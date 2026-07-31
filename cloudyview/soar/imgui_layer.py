"""Lazy ImGui/WGPU bridge used by the soar window app."""

from __future__ import annotations


class SoarImguiLayer:
    """Encode Dear ImGui draw data into an existing WGPU command encoder."""

    KEY_MAP_NAMES = {
        "ArrowDown": "down_arrow",
        "ArrowUp": "up_arrow",
        "ArrowLeft": "left_arrow",
        "ArrowRight": "right_arrow",
        "Backspace": "backspace",
        "CapsLock": "caps_lock",
        "Delete": "delete",
        "End": "end",
        "Enter": "enter",
        "Escape": "escape",
        "F1": "f1",
        "F2": "f2",
        "F3": "f3",
        "F4": "f4",
        "F5": "f5",
        "F6": "f6",
        "F7": "f7",
        "F8": "f8",
        "F9": "f9",
        "F10": "f10",
        "F11": "f11",
        "F12": "f12",
        "Home": "home",
        "Insert": "insert",
        "Alt": "left_alt",
        "Control": "left_ctrl",
        "Shift": "left_shift",
        "Meta": "left_super",
        "NumLock": "num_lock",
        "PageDown": "page_down",
        "PageUp": "page_up",
        "Pause": "pause",
        "PrintScreen": "print_screen",
        "ScrollLock": "scroll_lock",
        "Tab": "tab",
    }

    MOD_KEY_NAMES = {
        "Shift": ("mod_shift", "im_gui_mod_shift"),
        "Control": ("mod_ctrl", "im_gui_mod_ctrl"),
        "Alt": ("mod_alt", "im_gui_mod_alt"),
        "Meta": ("mod_super", "im_gui_mod_super"),
    }

    def __init__(self, *, device, target_format: str, canvas):
        try:
            from imgui_bundle import imgui
            from wgpu.utils.imgui import ImguiWgpuBackend
        except ImportError as e:  # pragma: no cover - env packaging only
            raise RuntimeError(
                "soar's in-window menu requires imgui-bundle plus "
                "wgpu.utils.imgui. Install the interactive extra after this "
                "change: uv sync --extra interactive"
            ) from e

        from .theme import SoarTheme

        self.imgui = imgui
        self._canvas = canvas
        self._context = imgui.create_context()
        imgui.set_current_context(self._context)
        self._backend = ImguiWgpuBackend(device, target_format)
        io = self._backend.io
        io.display_size = canvas.get_logical_size()
        scale = canvas.get_pixel_ratio()
        io.display_framebuffer_scale = (scale, scale)
        self.theme = SoarTheme(imgui)
        self._key_map = self._build_key_map()
        self._mod_key_map = self._build_mod_key_map()

    def _build_key_map(self):
        key_cls = self.imgui.Key
        key_map = {}
        for rendercanvas_key, attr in self.KEY_MAP_NAMES.items():
            if hasattr(key_cls, attr):
                key_map[rendercanvas_key] = getattr(key_cls, attr)
        return key_map

    def _build_mod_key_map(self):
        key_cls = self.imgui.Key
        key_map = {}
        for rendercanvas_key, attrs in self.MOD_KEY_NAMES.items():
            for attr in attrs:
                if hasattr(key_cls, attr):
                    key_map[rendercanvas_key] = getattr(key_cls, attr)
                    break
        return key_map

    def register_image(self, rgb: "object"):
        """Upload an (h, w, 3) uint8 image and return an ImGui texture ref.

        The caller owns the lifetime: hold the returned ``(ref, texture)``
        for as long as the image is drawn, then pass it to
        :meth:`release_image`. Textures registered here are unrelated to the
        font atlas the backend manages itself.
        """
        import numpy as np
        import wgpu

        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(
                f"register_image expects (h, w, 3|4) uint8; got {rgb.shape}."
            )
        height, width = rgb.shape[:2]
        if rgb.shape[2] == 3:
            rgba = np.empty((height, width, 4), dtype=np.uint8)
            rgba[..., :3] = rgb
            rgba[..., 3] = 255
        else:
            rgba = rgb

        device = self._backend._device
        texture = device.create_texture(
            label="imgui-image",
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            dimension="2d",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        device.queue.write_texture(
            {"texture": texture},
            rgba,
            {"bytes_per_row": width * 4, "rows_per_image": height},
            (width, height, 1),
        )
        view = texture.create_view()
        return self._backend.register_texture(view), (texture, view)

    def release_image(self, ref) -> None:
        """Drop a texture registered by :meth:`register_image`."""
        if ref is not None:
            self._backend.unregister_texture(ref)

    def handle_event(self, event: dict) -> None:
        """Forward rendercanvas events to ImGui IO."""
        io = self._backend.io
        etype = event["event_type"]
        if etype == "pointer_move":
            io.add_mouse_pos_event(event["x"], event["y"])
        elif etype in ("pointer_down", "pointer_up"):
            button = int(event.get("button", 1)) - 1
            io.add_mouse_button_event(button, etype == "pointer_down")
        elif etype == "wheel":
            io.add_mouse_wheel_event(
                event.get("dx", 0.0) / 100.0,
                -event.get("dy", 0.0) / 100.0,
            )
        elif etype in ("key_down", "key_up"):
            down = etype == "key_down"
            key_name = event["key"]
            key = self._imgui_key_for(key_name)
            if key is not None:
                io.add_key_event(key, down)
            mod_key = self._mod_key_map.get(key_name)
            if mod_key is not None:
                io.add_key_event(mod_key, down)
        elif etype == "char":
            text = event.get("data", "") or event.get("char_str", "") or ""
            if text:
                io.add_input_characters_utf8(text)

    def _imgui_key_for(self, key_name: str):
        mapped = self._key_map.get(key_name)
        if mapped is not None:
            return mapped
        if len(key_name) != 1:
            return None
        key_cls = self.imgui.Key
        ch = key_name.lower()
        if "0" <= ch <= "9" and hasattr(key_cls, "_0"):
            return key_cls(key_cls._0.value + (ord(ch) - ord("0")))
        if "a" <= ch <= "z" and hasattr(key_cls, "a"):
            return key_cls(key_cls.a.value + (ord(ch) - ord("a")))
        return None

    def encode(self, command_encoder, target_view, update_gui) -> None:
        """Build and render one ImGui frame over ``target_view``."""
        import wgpu

        imgui = self.imgui
        imgui.set_current_context(self._context)
        io = self._backend.io
        io.display_size = self._canvas.get_logical_size()
        scale = self._canvas.get_pixel_ratio()
        io.display_framebuffer_scale = (scale, scale)

        imgui.new_frame()
        try:
            update_gui(imgui)
        finally:
            imgui.render()

        render_pass = command_encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "resolve_target": None,
            "clear_value": (0, 0, 0, 1),
            "load_op": wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
        }])
        self._backend.render(imgui.get_draw_data(), render_pass)
        render_pass.end()
