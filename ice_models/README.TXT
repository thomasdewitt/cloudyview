
These netcdf files should contain all the properties that you'll need to perform radiative 
transfer calculations. Each file contains ice models for 445 wavelengths from 0.2 to 
100 microns, each of which contain single-scattering properties as a function of 
effective diameter that ranges from 10 to 120 microns in increments of 5 microns (for a 
total of 23 effective diameters).  The gzip'd files contain the full phase matrix and are
so named. The ".nc" files do not contain the full phase matrix and are labeled 
"NoPhaseMatrix" - just to be as clear as possible on this point.

We are supplying models based on our general habit mixture (GHM), use of solid
 columns only, and use of the aggregate of solid columns only. Each of these sets of 
models assume that the particles are randomly oriented and also assume severe 
surface roughening.

The phase matrix includes P11, P21/P11, P22/P11, P33/P11, P43/P11, and P44/P11, 
with each defined at 498 scattering angles. The scattering angles should also be 
provided as a separate parameter.

I am in the midst of writing a short article describing updates from my 2011 paper on 
this topic, which was only for shortwave models. I can't quite seem to complete this 
paper, so in the meantime, please refer to the 2011 paper for details - the 
methodology has not changed in the meantime. 

Let me know if you have questions. If you find any issues, let me know and I will try to 
correct them. I've sent these files out to tire-kickers in the past and have 
incorporated suggestions that I've received.

I am also providing two matlab routines that I used to generate some figures of 
properties and their differences between models - hope you find these useful. One 
routine plots single scattering properties (SSP) in the shortwave (SW); the other in the 
longwave (LW). These matlab files have a ".m" extension.

Also added a matlab routine to plot the scattering phase function (P11) for three
effective diameters for each habit model, called plot_P11_3HabitMixtures_OneWavelength.m.

Hope these models are useful -

No worries,
Bryan

Last Updated: 25 July 2013