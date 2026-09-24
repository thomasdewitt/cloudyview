//! regcount: what the NVIDIA Vulkan driver compiled a WGSL entry point into.
//!
//!   regcount <file.wgsl> <fragment-entry> <rgba16f|rgba32f|r16f> [vertex-entry]
//!
//! WGSL -> SPIR-V with naga (the compiler wgpu-native uses), a graphics
//! pipeline built with VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR, and the
//! driver's own statistics per executable (VK_KHR_pipeline_executable_
//! properties) printed: on NVIDIA that includes the register count, which is
//! what decides how many warps a shader can keep in flight.
use ash::vk;
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: regcount <file.wgsl> <fragment-entry> <rgba16f|rgba32f|r16f> [vertex-entry]");
        std::process::exit(2);
    }
    let src = std::fs::read_to_string(&args[1]).expect("read wgsl");
    let frag_entry = args[2].clone();
    let format = match args[3].as_str() {
        "rgba16f" => vk::Format::R16G16B16A16_SFLOAT,
        "rgba32f" => vk::Format::R32G32B32A32_SFLOAT,
        "r16f" => vk::Format::R16_SFLOAT,
        other => panic!("unknown format {other}"),
    };
    let vert_entry = args.get(4).cloned().unwrap_or_else(|| "vs_main".into());

    // --- WGSL -> SPIR-V ------------------------------------------------------
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
    let mut opts = naga::back::spv::Options::default();
    opts.lang_version = (1, 3);
    opts.bounds_check_policies = naga::proc::BoundsCheckPolicies {
        index: naga::proc::BoundsCheckPolicy::Restrict,
        buffer: naga::proc::BoundsCheckPolicy::Restrict,
        image_load: naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
        binding_array: naga::proc::BoundsCheckPolicy::Restrict,
    };
    let spv = naga::back::spv::write_vec(&module, &info, &opts, None).expect("spv");

    // --- descriptor layouts from the module's own bindings -------------------
    let mut groups: BTreeMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = BTreeMap::new();
    for (_, var) in module.global_variables.iter() {
        let Some(rb) = &var.binding else { continue };
        let inner = &module.types[var.ty].inner;
        let ty = match var.space {
            naga::AddressSpace::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
            naga::AddressSpace::Storage { .. } => vk::DescriptorType::STORAGE_BUFFER,
            naga::AddressSpace::Handle => match inner {
                naga::TypeInner::Image { class, .. } => match class {
                    naga::ImageClass::Storage { .. } => vk::DescriptorType::STORAGE_IMAGE,
                    _ => vk::DescriptorType::SAMPLED_IMAGE,
                },
                naga::TypeInner::Sampler { .. } => vk::DescriptorType::SAMPLER,
                other => panic!("unhandled handle type {other:?}"),
            },
            other => panic!("unhandled address space {other:?}"),
        };
        groups.entry(rb.group).or_default().push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(rb.binding)
                .descriptor_type(ty)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE),
        );
    }

    unsafe {
        let entry = ash::Entry::load().expect("vulkan loader");
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
            .expect("instance");
        let pdev = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .find(|&p| {
                instance.get_physical_device_properties(p).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .expect("no discrete GPU");
        let pname = CStr::from_ptr(instance.get_physical_device_properties(pdev).device_name.as_ptr())
            .to_string_lossy()
            .to_string();
        let qfam = instance
            .get_physical_device_queue_family_properties(pdev)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap() as u32;
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfam).queue_priorities(&prio)];
        let exts = [ash::khr::pipeline_executable_properties::NAME.as_ptr()];
        let mut pe_feat = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&exts)
            .push_next(&mut pe_feat)
            .push_next(&mut f13);
        let device = instance.create_device(pdev, &dci, None).expect("device");

        let max_group = groups.keys().copied().max().unwrap_or(0);
        let mut set_layouts = Vec::new();
        for g in 0..=max_group {
            let empty = Vec::new();
            let b = groups.get(&g).unwrap_or(&empty);
            set_layouts.push(
                device
                    .create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(b), None)
                    .unwrap(),
            );
        }
        let layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts), None)
            .unwrap();
        let sm = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
            .expect("shader module");
        let vname = CString::new(vert_entry.clone()).unwrap();
        let fname = CString::new(frag_entry.clone()).unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(sm).name(&vname),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(sm).name(&fname),
        ];
        let vi = vk::PipelineVertexInputStateCreateInfo::default();
        let ia = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let vp = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rs = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let ms = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let att = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
        let cb = vk::PipelineColorBlendStateCreateInfo::default().attachments(&att);
        let dyns = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let ds = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyns);
        let fmts = [format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&fmts);
        let t0 = std::time::Instant::now();
        let gpci = vk::GraphicsPipelineCreateInfo::default()
            .flags(vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR)
            .stages(&stages)
            .vertex_input_state(&vi)
            .input_assembly_state(&ia)
            .viewport_state(&vp)
            .rasterization_state(&rs)
            .multisample_state(&ms)
            .color_blend_state(&cb)
            .dynamic_state(&ds)
            .layout(layout)
            .push_next(&mut rendering);
        let pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[gpci], None)
            .map_err(|(_, e)| e)
            .expect("pipeline")[0];
        let compile_ms = t0.elapsed().as_secs_f64() * 1e3;

        let pe = ash::khr::pipeline_executable_properties::Device::new(&instance, &device);
        let pinfo = vk::PipelineInfoKHR::default().pipeline(pipeline);
        let props = pe.get_pipeline_executable_properties(&pinfo).expect("props");
        println!("device: {pname}; entry {frag_entry}; spirv {} words; pipeline compile {compile_ms:.0} ms",
                 spv.len());
        for (i, p) in props.iter().enumerate() {
            let name = CStr::from_ptr(p.name.as_ptr()).to_string_lossy();
            println!("executable {i}: {name} ({:?}), subgroup {}", p.stages, p.subgroup_size);
            let einfo = vk::PipelineExecutableInfoKHR::default().pipeline(pipeline).executable_index(i as u32);
            let stats = pe.get_pipeline_executable_statistics(&einfo).expect("stats");
            for s in stats {
                let sname = CStr::from_ptr(s.name.as_ptr()).to_string_lossy();
                let v = match s.format {
                    vk::PipelineExecutableStatisticFormatKHR::BOOL32 => format!("{}", s.value.b32),
                    vk::PipelineExecutableStatisticFormatKHR::INT64 => format!("{}", s.value.i64),
                    vk::PipelineExecutableStatisticFormatKHR::UINT64 => format!("{}", s.value.u64),
                    vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => format!("{}", s.value.f64),
                    _ => "?".into(),
                };
                println!("    {sname:32} {v}");
            }
        }
        device.destroy_pipeline(pipeline, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
