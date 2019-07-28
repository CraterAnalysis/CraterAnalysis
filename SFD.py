"""
Created on Fri Mar 11 16:00:00 2016

@author: Stuart J. Robbins

This code will create a size-frequency distribution (SFD) using a kernel density
estimator (KDE), which produces a probability distribution function (PDF). It
models each crater diameter as a Gaussian kernel. It uses a bootstrap sampling
with replacement to calculate confidence intervals. It will both display graphs
of each traditional SFD type (cumulative, relative, differential, incremental)
and output a .CSV table that the user can import into their graphing software of
choice. For more information, see: Robbins et al. (2018) "Revised recommended
methods for analyzing crater size-frequency distributions." Meteoritics &
Planetary Science, 53(4), 891-931. doi: 10.1111/maps.12990.

"""

##TO DO LIST:
#General Software Usability / Options:
#  - Convert all user-set variables to command-line options, and have warnings
#    for when things are incompatible.
#  - Support for more than one set of diameters, especially for display output.
#  - Support for power-law fitting and graphing, both small-end truncated Pareto
#    and large-end.  Output confidences, uncertainty envelopes, and the mini-PDF
#    showing that uncertainty model as introduced by Michael (2016).
#  - Support for fitting Neukum and Hartmann production functions.
#  - Support for user input of file name that has the crater diameters.
#
#KDE:
#  - Multi-thread if desired (it is already very fast for 10s of thousands of
#    craters.
#  - Implement alternative kernals that are controlled with a command-line arg.
#
#Confidence Interval:
#  - Multi-thread.
#  - Implement alternative kernals that are controlled with a command-line arg.
#
#CSV Output:
#  - Allow user to specify file name (control with command-line arg).
#
#Graph Output:
#  - Allow setting of colors.
#  - Proper scaling of axes.
#  - Fix overlapping text due to default Python display of axes tick labels.


import sys
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import scipy.stats as spStats
import scipy.special as spSpecial

np.set_printoptions(threshold=sys.maxsize) #for debugging purposes




def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


##----------------------------------------------------------------------------##



##----- USER-MODIFIED VARIABLES -----##

d_DIAM_min                  = 0.03125   #what is the minimum CEIL(diameter) bin - note that this will be smaller in the end due to bin centering
d_apriori_uncertainty       = 0.1       #a priori uncertainty on diameter as a multiple of the diameter
d_apriori_Nvariation        = 0.0		#a priori variation in number of craters found as a strict fraction of the total (e.g., "0.3" = +/-30%)
d_descretization            = 0.01      #fidelity of the descritization in km (or whatever unit the crater diameters are in)
d_sigma_range               = 3.0       #range will be min(diameter) - d_sigma_range*min(diameter)'s uncertainty  --to--  d_sigma_range*max(diameter)'s uncertainty + max(diameter)
d_sigma_range_min           = 5         #range of KDE starts at min(diameter) - d_sigma_range*min(diameter)'s uncertainty
d_sigma_range_max           = 5         #range of KDE ends at d_sigma_range*max(diameter)'s uncertainty + max(diameter)
d_sigma_multiplier_4conf    = 1         #bootstrap samples original diameters if difference between adjacent diameters is < THIS VARIABLE * sum of the adjacent diameter uncertainties
i_MonteCarlo_conf_max       = 10000      #number of runs to do for confidence bands
i_MonteCarlo_conf_interval  = 100       #number of runs to do for confidence bands before checking for convergence
d_MonteCarlo_conf_converge  = 1e-4      #tolerance for confidence band convergence
d_confidence                = 2         #sigmas of confidence band
d_level_for_completeness    = 0.8       #completeness estimate finds the "level" steady-state for the first derivative of the DSDF, when the value is THIS fration of it, that's completeness
f_output_time_diagnostics   = 1         #0 = no, 1 = yes
f_MT                        = 0         #0 = no to multi-thread the bootstrap; 1 = yes to do the multi-thread
d_surfacearea               = 1         #surface area for normalization



##----- NOTHING BELOW HERE SHOULD BE MODIFIED -----##

#Store the time to output it at the end.
timer_start = time.time()
tt = time.time()

#Output some information to the user.
print("This code factors in:\n\t±%g%% uncertainty on crater diameter\n\t±%g%% uncertainty on number of craters found\n\t%d Monte Carlo runs for the bootstrap-based uncertainty at %d-sigma (±%g%%)\n" % (d_apriori_uncertainty*100, d_apriori_Nvariation*100,i_MonteCarlo_conf_max,d_confidence,spSpecial.erf(d_confidence/math.sqrt(2))*100))



##----- SET UP THE DATA AND IMPORTANT VARIABLES AND VECTORS/ARRAYS -----##


#Read the crater data into the array DIAM_TEMP and DIAM_TEMP_SD.  The read-in
# via numpy removes any NaN values (missing rows), so we do not need to check..
DIAM_TEMP = np.genfromtxt('craters.csv',delimiter=',')
DIAM_TEMP = np.resize(DIAM_TEMP,(len(DIAM_TEMP),1))

#Check to see if the minimum diameter (d_DIAM_start) is > the minimum diameter
# crater.  If it is, need to remove those smaller.
if min(DIAM_TEMP) < d_DIAM_min:
    DIAM_TEMP = DIAM_TEMP[DIAM_TEMP > d_DIAM_min]
    print("Removing craters smaller than the d_DIAM_min variable.")

#Sort the diameters (will save time later).
DIAM_TEMP.sort()

#Assign a priori uncertainty.
DIAM_SD_TEMP = d_apriori_uncertainty*DIAM_TEMP      #does not yet support custom uncertainties
if(f_output_time_diagnostics):
    print("Time to read the data: %g seconds" % (time.time()-tt))
    tt=time.time()


#Now that we're done setting up the craters, set other important variables.
# The min and max here will be the min/max diameters to which the KDE
# KDE calculation is carried out.  The total craters saves a tiny amount of time
# so the code doesn't have to keep calculating it.
d_DIAM_min      = max(d_DIAM_min,(min(DIAM_TEMP)-d_sigma_range*min(DIAM_SD_TEMP)))
d_DIAM_max      = max(DIAM_TEMP)+d_sigma_range*max(DIAM_SD_TEMP)
i_total_craters = len(DIAM_TEMP)


#Determine how many "bins" there will be based on the d_DIAM_min, the largest
# crater, and the descritization variable.
i_number_kde    = int(math.floor(math.log10(d_DIAM_max/d_DIAM_min)/d_descretization))

#Declare all the necessary arrays and initialize them.  (Don't worry, we'll change
# the names later.)
edf_diam                = np.power(10,(np.log10(d_DIAM_min)+np.arange(0,i_number_kde+1,1)*d_descretization))
edf_csfd                = [0]*(i_number_kde+1)
edf_isfd                = [0]*(i_number_kde+1)
edf_dsfd                = [0]*(i_number_kde+1)
edf_rsfd                = [0]*(i_number_kde+1)
edf_erro_csfd_pos_1s    = [0]*(i_number_kde+1)
edf_erro_csfd_pos       = [0]*(i_number_kde+1)
edf_erro_csfd_neg_1s    = [0]*(i_number_kde+1)
edf_erro_csfd_neg       = [0]*(i_number_kde+1)
edf_erro_isfd_pos_1s    = [0]*(i_number_kde+1)
edf_erro_isfd_pos       = [0]*(i_number_kde+1)
edf_erro_isfd_neg_1s    = [0]*(i_number_kde+1)
edf_erro_isfd_neg       = [0]*(i_number_kde+1)
edf_erro_dsfd_pos_1s    = [0]*(i_number_kde+1)
edf_erro_dsfd_pos       = [0]*(i_number_kde+1)
edf_erro_dsfd_neg_1s    = [0]*(i_number_kde+1)
edf_erro_dsfd_neg       = [0]*(i_number_kde+1)
edf_erro_rsfd_pos_1s    = [0]*(i_number_kde+1)
edf_erro_rsfd_pos       = [0]*(i_number_kde+1)
edf_erro_rsfd_neg_1s    = [0]*(i_number_kde+1)
edf_erro_rsfd_neg       = [0]*(i_number_kde+1)



##----- CREATE THE KDE IN DIFFERENTIAL FORMAT -----##


#The actual main KDE algorithm, filling the DIFFERENTIAL size-frequency data.
# This is a kernel density function, meaning that each discrete diameter point
# is treated as a Gaussian distribution.  These are then summed to create the
# kernel.
if(f_output_time_diagnostics):
    print("Time to create all the vectors and set diameter wave: %g seconds" % (tt-timer_start))

##Method 1: Maybe slightly faster than Method 2.  This works by going crater-by-
## crater, determining the diameter range overwhich it will have any influence on
## the kernel, and only doing the math for that diameter range.  Based on tests
## up to ~20,000 craters, it is comparable in speed but maybe 1% faster than the
## next method.
#for i_craterIndex_current in range(0,i_total_craters,1):
#    start = int(np.searchsorted(edf_diam, DIAM_TEMP[i_craterIndex_current]-DIAM_SD_TEMP[i_craterIndex_current]*d_sigma_range))
#    stop  = int(np.searchsorted(edf_diam, DIAM_TEMP[i_craterIndex_current]+DIAM_SD_TEMP[i_craterIndex_current]*d_sigma_range))
#
#    #Gaussian
#    edf_dsfd[start:stop] += (1./ (i_total_craters*DIAM_SD_TEMP[i_craterIndex_current]*2))*np.exp(-0.5*((edf_diam[start:stop]-DIAM_TEMP[i_craterIndex_current])/DIAM_SD_TEMP[i_craterIndex_current]*(edf_diam[start:stop]-DIAM_TEMP[i_craterIndex_current])/DIAM_SD_TEMP[i_craterIndex_current]))

#Method 2: This method goes diameter-by-diameter and sums the kernel contribu-
# tions for every crater for that diameter.  It is likely comparable to Method 1
# only in Python due to numpy optimizations in not doing math when values would
# be very close to zero.  But I am speculating.  The take-home message is that
# if you port this to another language, Method 1 may be significantly faster or
# slower than Method 2, depending.
for i_diameterIndex_current in range(0,int(i_number_kde)+1,1):
    #Gaussian
    edf_dsfd[i_diameterIndex_current] = np.sum( np.reciprocal(i_total_craters*DIAM_SD_TEMP[:]*2)*np.exp(-0.5*((edf_diam[i_diameterIndex_current]-DIAM_TEMP[:])/DIAM_SD_TEMP[:])*((edf_diam[i_diameterIndex_current]-DIAM_TEMP[:])/DIAM_SD_TEMP[:])) )

if(f_output_time_diagnostics):
    print("Time to create the empirical density function: %g seconds" % (time.time()-tt))
    tt=time.time()



##----- CREATE THE ALTERNATIVE SFD DISPLAY TYPES -----##


#Divide the differential plot by diameter^-3 to make the R-plot.
edf_rsfd = edf_dsfd[:] * np.power(3,edf_diam[:])

#The definition of the differential is (csfd[i]-csfd[i+1]) / (diam[i+1]-diam[i])
# So to make the incremental, from which we make the cumulative, the incremental
# is (dsfd[i]+dsfd[i+1])/2 * (diam[i+1]-diam[i]).
edf_isfd[:] = edf_dsfd[:]
for i_counter in range(0, len(edf_isfd)-2): #there is likely a more Python way to do this
    edf_isfd[i_counter] = (edf_isfd[i_counter]+edf_isfd[i_counter+1])/2 * (edf_diam[i_counter+1]-edf_diam[i_counter])

#"Integrate" the incremental plot to make the cumulative.
edf_csfd[len(edf_csfd)-1] = edf_isfd[len(edf_isfd)-1]
for i_counter in range(0, len(edf_isfd)-1, 1):
    edf_csfd[i_counter] = np.sum(edf_isfd[i_counter:len(edf_csfd)])

#Store NaN to the ultimate value of the DSFD.
edf_dsfd[len(edf_dsfd)-1] = float('NaN')

#Remove the last point of the incremental.
#edf_isfd[0]                                 = float('NaN')
edf_isfd[len(edf_isfd)-1]                   = float('NaN')
#edf_erro_isfd_pos[0]                        = float('NaN')
#edf_erro_isfd_neg[0]                        = float('NaN')
edf_erro_isfd_pos[len(edf_erro_isfd_pos)-1] = float('NaN')
edf_erro_isfd_neg[len(edf_erro_isfd_neg)-1] = float('NaN')
#edf_csfd[0]                                 = float('NaN')
#edf_csfd[len(edf_csfd)-1]                   = float('NaN')
#edf_erro_csfd_pos[0]                        = float('NaN')
#edf_erro_csfd_neg[0]                        = float('NaN')
#edf_erro_csfd_pos[len(edf_erro_csfd_pos)-1] = float('NaN')
#edf_erro_csfd_neg[len(edf_erro_csfd_neg)-1] = float('NaN')



##----- CREATE THE CONFIDENCE BANDS -----##

#Bootstrap with resampling to determine confidence intervals.  We have to do
# this after creation of the cumulative because a method to do the bootstrap is
# to sample from the PDF (KDE makes the PDF), and if the PDF is an EDF, it needs
# a cumulative from which to sample.  However, because the cumulative needs to
# be scaled to 1, we need to duplicate it so the scaling is not messed up later.

#Create the 1-CDF for bootstrapping.
edf_cedf = edf_csfd.copy()
dummy_scale = edf_csfd[0]
edf_cedf   /= dummy_scale


#Bootstrap with resampling to determine confidence intervals.
bootstrap_data = np.zeros((i_number_kde+1,i_MonteCarlo_conf_max))
bootstrap_test_converge = np.zeros((i_number_kde+1,math.ceil(i_MonteCarlo_conf_max/i_MonteCarlo_conf_interval)-1))
if i_MonteCarlo_conf_max > 0:
    if f_MT == 0:
        i_counter1 = 0
        while True:
#        for i_counter1 in range (0, i_MonteCarlo_conf_max, 1):
            current_bootstrap_iterations = int(np.round(i_total_craters*(np.random.uniform(-1,1)*d_apriori_Nvariation+1)))  #so we can vary number of points per bootstrap
            for i_counter2 in range (0, current_bootstrap_iterations, 1):

                sample_diam = [0]*(current_bootstrap_iterations) #initialize vector
                sample_sd   = [0]*(current_bootstrap_iterations) #initialize vector
                
                #Bootstrap sampling from the original diameters.
                #sample_loc  = [int(np.random.uniform(0,i_total_craters-1)) for i in range(i_total_craters-1)] #initialize vector with random numbers
                #sample_loc  = np.array(sample_loc).reshape(i_total_craters-1) #convert to numpy and lop off one dimension
                #sample_diam = DIAM_TEMP[sample_loc]
                #sample_sd   = DIAM_SD_TEMP[sample_loc]
                
                #Smoothed bootstrap sampling from the original diameters.
                #sample_loc  = [int(np.random.uniform(0,i_total_craters-1)) for i in range(i_total_craters-1)] #initialize vector with random numbers
                #sample_loc  = np.array(sample_loc).reshape(i_total_craters-1) #convert to numpy and lop off one dimension
                #sample_sd   = DIAM_SD_TEMP[sample_loc]
                #sample_diam = DIAM_TEMP[sample_loc]+np.random.normal(0,sample_sd)
                
                #Smoothed bootstrap sampling from the KDE itself.
                #sample_loc   = [np.random.uniform(0,1) for i in range(current_bootstrap_iterations)] #initialize vector with random numbers
                #for i_counter2 in range (0,int(current_bootstrap_iterations)):
                #    location = find_nearest(edf_cedf,sample_loc[i_counter2])
                #    sample_diam[i_counter2] = edf_diam[location]
                #sample_sd    = np.asarray(sample_diam) * d_apriori_uncertainty

                #Hybrid between smoothed and direct sampling.
                sample_loc  = min(round((np.random.uniform(0,1))*i_total_craters),i_total_craters-1) #create a uniformly distributed random variate in [0,1]
                test_diam   = DIAM_TEMP[round(sample_loc)]              #find the random diameter, per method 1
                if sample_loc <= 0.5:                                   #first part of hybrid, special case for first value
                    sample_diam = DIAM_TEMP[0]                          #if we're at or before the first point, use the first real diameter point (avoid spending power where we don't have real craters)
                elif sample_loc >= i_total_craters-2:                   #second part, special case for the largest diameters to follow the EDF
                    sample_diam = edf_diam[int(np.searchsorted(edf_diam,DIAM_TEMP[sample_loc]))]
                elif ((DIAM_TEMP[round(sample_loc)]-DIAM_TEMP[round(sample_loc)-1]) < d_sigma_multiplier_4conf*d_apriori_uncertainty*(DIAM_TEMP[round(sample_loc)]+DIAM_TEMP[round(sample_loc)-1])) and ((DIAM_TEMP[round(sample_loc)+1]-DIAM_TEMP[round(sample_loc)]) < d_sigma_multiplier_4conf*d_apriori_uncertainty*(DIAM_TEMP[round(sample_loc)+1]+DIAM_TEMP[round(sample_loc)])):
                    sample_diam = test_diam                             #third part, if we're in a "dense" region of diameters, use the original diameters
                else:                                                   #fourth part, when we are in a sparse region and want to sample the EDF
                    sample_diam = edf_diam[int(np.searchsorted(edf_diam,DIAM_TEMP[sample_loc]))]
                sample_sd   = sample_diam * d_apriori_uncertainty
                
                #Limit the calculation to a certain range to save time for
                # distributions in the domain (-inf,+inf).
                start =     math.floor(np.searchsorted(edf_diam, sample_diam-sample_sd*d_sigma_range))
                stop  = min(math.ceil (np.searchsorted(edf_diam, sample_diam+sample_sd*d_sigma_range)),i_number_kde)
                
                #Gaussian
                for dummy_location in range(start, stop+1, 1): #could not figure out how to get this to work in Python notation
                    bootstrap_data[dummy_location,i_counter1] += (1/(i_total_craters*sample_sd*2))*math.exp(-0.5*((edf_diam[dummy_location]-sample_diam)/sample_sd)*((edf_diam[dummy_location]-sample_diam)/sample_sd))
            

            #Calculate the current confidence intervals to check for converg-
            # ance.  Only doing the positive one to avoid lots of zeros at the
            # large diameters.
            if(((i_counter1 % i_MonteCarlo_conf_interval) == 0) and (i_counter1 > 0)):
                for i_counter3 in range(0, i_number_kde+1, 1):
                    #For this discretized diameter location, extract the data
                    # from the bootstrap.  Sort it to calculate CIs.
                    dummy = np.zeros(i_counter1)
                    dummy = bootstrap_data[i_counter3][:i_counter1].copy()
                    dummy.sort()

                    #Find the DSFD value in the sorted bootstrap list.
                    V_LevelX = int(np.searchsorted(dummy,edf_dsfd[i_counter3]))
                    
                    #Store to our convergence matrix the difference between the
                    # DSFD value and the +CI from the latest bootstrap for this
                    # discretized diameter location.  Why?  Because our test for
                    # convergence is whether the average of this difference,
                    # across all diameter locations, is less than our threshold.
                    bootstrap_test_converge[i_counter3, int(i_counter1/i_MonteCarlo_conf_interval-1)] = abs( edf_dsfd[i_counter3]-dummy[min(int(V_LevelX+(i_counter1-V_LevelX)*spSpecial.erf(d_confidence/math.sqrt(2))),i_counter1-1)] )
#                    if i_counter3 == 10: print(bootstrap_data[10,:i_counter1],dummy[int(V_LevelX+(i_counter1-V_LevelX)*spSpecial.erf(d_confidence/math.sqrt(2)))])
#                print(bootstrap_test_converge)
                #Since the point of convergence is something hasn't changed, we
                # can only test after having done this at least twice.
                if i_counter1/i_MonteCarlo_conf_interval >= 2:
                    #Initialize.
                    bootstrap_test_converge_dsfd = [0]*(i_number_kde+1)
                    
                    #Calculate the difference between the most recent values in
                    # bootstrap_test_converge and the one before it, and then
                    # normalize by the most recent.  We normalize to get the
                    # fraction difference, rather than an absolute difference.
                    bootstrap_test_converge_dsfd = [abs( bootstrap_test_converge[counter][int(i_counter1/i_MonteCarlo_conf_interval-1)] - bootstrap_test_converge[counter][int(i_counter1/i_MonteCarlo_conf_interval-2)] ) / bootstrap_test_converge[counter][int(i_counter1/i_MonteCarlo_conf_interval-1)] for counter in range(0,i_number_kde)]
#                    print(bootstrap_test_converge_dsfd)

                    #Now do the testing.  Need to avoid NaNs for numpy, so have
                    # to specifically index within the averaging.
                    print(np.average(bootstrap_test_converge_dsfd[0:int(i_counter1/i_MonteCarlo_conf_interval)]))#, bootstrap_data[5,i_counter1], bootstrap_data[5,i_counter1-1], bootstrap_data[5,i_counter1]-bootstrap_data[5,i_counter1-1])
                    if(np.average(bootstrap_test_converge_dsfd[0:int(i_counter1/i_MonteCarlo_conf_interval)]) <= d_MonteCarlo_conf_converge):
                        i_MonteCarlo_conf_max = i_counter1
                        f_MonteCarlo_converge = 1
                        print("Bootstrap converged to %e after %g iterations.\n" % (d_MonteCarlo_conf_converge, i_MonteCarlo_conf_max))

            #Increment the counter through the Monte Carlo iterations and test
            # for being done.
            i_counter1 += 1
            if i_counter1 >= i_MonteCarlo_conf_max:
                break



    else:
        #---multi-threading not yet coded!!---#
        BootstrapMTPrepare(DIAM_TEMP,DIAM_SD_TEMP,edf_diam,bootstrap_data,i_number_kde,i_MonteCarlo_conf_max,i_total_craters)

    if(f_output_time_diagnostics):
        print("Time to bootstrap to create the confidence interval: %g seconds" % (time.time()-tt))
        tt=time.time()
    
    for i_counter1 in range (0,i_number_kde+1):
        dummy = np.zeros(i_MonteCarlo_conf_max)
        dummy[:] = bootstrap_data[i_counter1][:len(dummy)] #in case we truncated due to convergence!
        dummy.sort()
        level = find_nearest(dummy,edf_dsfd[i_counter1])
        #edf_erro_dsfd_neg[i_counter1] = np.abs( edf_dsfd[i_counter1]-dummy[level*(1-spSpecial.erf(d_confidence/np.sqrt(2.)))] )
        #edf_erro_dsfd_pos[i_counter1] = np.abs( edf_dsfd[i_counter1]+dummy[level+(i_MonteCarlo_conf_max-level)*spSpecial.erf(d_confidence/np.sqrt(2.))] )
        edf_erro_dsfd_neg[i_counter1]    = dummy[int(level*(1-spSpecial.erf(d_confidence/np.sqrt(2.))))]
        edf_erro_dsfd_neg_1s[i_counter1] = dummy[int(level*(1-spSpecial.erf(1.          /np.sqrt(2.))))]
        edf_erro_dsfd_pos[i_counter1]    = dummy[int(level+(i_MonteCarlo_conf_max-level)*spSpecial.erf(d_confidence/np.sqrt(2.)))]
        edf_erro_dsfd_pos_1s[i_counter1] = dummy[int(level+(i_MonteCarlo_conf_max-level)*spSpecial.erf(1.          /np.sqrt(2.)))]
    if(f_output_time_diagnostics):
        print("Time to create the confidence interval from the bootstrapped data: %g seconds" % (time.time()-tt))
        tt=time.time()
#No idea why, but for some reason these get reshaped to be 100x their should-be size.
edf_diam = edf_diam.reshape(i_number_kde+1)


#The kernel density plot needs to be scaled to get it into real numbers.  Figure
# out that scaling and apply it to both the kernel and the cumulative.
d_scaling = len(DIAM_TEMP)/edf_csfd[0]/d_surfacearea
edf_dsfd                = np.array(edf_dsfd)*d_scaling
edf_erro_dsfd_neg       = np.array(edf_erro_dsfd_neg)*d_scaling
edf_erro_dsfd_pos       = np.array(edf_erro_dsfd_pos)*d_scaling
edf_erro_dsfd_neg_1s    = np.array(edf_erro_dsfd_neg_1s)*d_scaling
edf_erro_dsfd_pos_1s    = np.array(edf_erro_dsfd_pos_1s)*d_scaling
edf_rsfd                = np.array(edf_rsfd)*d_scaling
edf_erro_rsfd_neg       = np.multiply(np.divide(edf_erro_dsfd_neg,edf_dsfd),edf_rsfd)
edf_erro_rsfd_pos       = np.multiply(np.divide(edf_erro_dsfd_pos,edf_dsfd),edf_rsfd)
edf_erro_rsfd_neg_1s    = np.multiply(np.divide(edf_erro_dsfd_neg_1s,edf_dsfd),edf_rsfd)
edf_erro_rsfd_pos_1s    = np.multiply(np.divide(edf_erro_dsfd_pos_1s,edf_dsfd),edf_rsfd)
edf_isfd                = np.array(edf_isfd)*d_scaling
edf_erro_isfd_neg       = np.multiply(np.divide(edf_erro_dsfd_neg,edf_dsfd),edf_isfd)
edf_erro_isfd_pos       = np.multiply(np.divide(edf_erro_dsfd_pos,edf_dsfd),edf_isfd)
edf_erro_isfd_neg_1s    = np.multiply(np.divide(edf_erro_dsfd_neg_1s,edf_dsfd),edf_isfd)
edf_erro_isfd_pos_1s    = np.multiply(np.divide(edf_erro_dsfd_pos_1s,edf_dsfd),edf_isfd)
edf_csfd                = np.array(edf_csfd)*d_scaling
edf_erro_csfd_neg       = np.multiply(np.divide(edf_erro_dsfd_neg,edf_dsfd),edf_csfd)
edf_erro_csfd_pos       = np.multiply(np.divide(edf_erro_dsfd_pos,edf_dsfd),edf_csfd)
edf_erro_csfd_neg_1s    = np.multiply(np.divide(edf_erro_dsfd_neg_1s,edf_dsfd),edf_csfd)
edf_erro_csfd_pos_1s    = np.multiply(np.divide(edf_erro_dsfd_pos_1s,edf_dsfd),edf_csfd)
if(f_output_time_diagnostics):
    print("Time to create all the other plots and scale: %g seconds" % (time.time()-tt))
    tt=time.time()



#Output the time.
print("Creating the SFD took %g seconds." % (time.time()-timer_start))



##----- WRITE OUT A LARGE CSV TABLE SO THE USER CAN GRAPH ON THEIR OWN -----##
fOut = open('output.csv','w')
fOut.write("edf_diam,edf_erro_csfd_neg,edf_erro_csfd_neg_1s,edf_csfd,edf_erro_csfd_pos_1s,edf_erro_csfd_pos,edf_erro_rsfd_neg,edf_erro_rsfd_neg_1s,edf_rsfd,edf_erro_rsfd_pos_1s,edf_erro_rsfd_pos,edf_erro_dsfd_neg,edf_erro_dsfd_neg_1s,edf_dsfd,edf_erro_dsfd_pos_1s,edf_erro_dsfd_pos,edf_erro_isfd_neg,edf_erro_isfd_neg_1s,edf_isfd,edf_erro_isfd_pos_1s,edf_erro_isfd_pos\n")
for counterPoint in range(0,len(edf_diam)-1,1):
    outputstring = str(edf_diam[counterPoint]) + ","
    outputstring += str(edf_erro_csfd_neg[counterPoint]) + ","
    outputstring += str(edf_erro_csfd_neg_1s[counterPoint]) + ","
    outputstring += str(edf_csfd[counterPoint]) + ","
    outputstring += str(edf_erro_csfd_pos_1s[counterPoint]) + ","
    outputstring += str(edf_erro_csfd_pos[counterPoint]) + ","
    outputstring += str(edf_erro_rsfd_neg[counterPoint]) + ","
    outputstring += str(edf_erro_rsfd_neg_1s[counterPoint]) + ","
    outputstring += str(edf_rsfd[counterPoint]) + ","
    outputstring += str(edf_erro_rsfd_pos_1s[counterPoint]) + ","
    outputstring += str(edf_erro_rsfd_pos[counterPoint]) + ","
    outputstring += str(edf_erro_dsfd_neg[counterPoint]) + ","
    outputstring += str(edf_erro_dsfd_neg_1s[counterPoint]) + ","
    outputstring += str(edf_dsfd[counterPoint]) + ","
    outputstring += str(edf_erro_dsfd_pos_1s[counterPoint]) + ","
    outputstring += str(edf_erro_dsfd_pos[counterPoint]) + ","
    outputstring += str(edf_erro_isfd_neg[counterPoint]) + ","
    outputstring += str(edf_erro_isfd_neg_1s[counterPoint]) + ","
    outputstring += str(edf_isfd[counterPoint]) + ","
    outputstring += str(edf_erro_isfd_pos_1s[counterPoint]) + ","
    outputstring += str(edf_erro_isfd_pos[counterPoint]) + "\n"
    fOut.write(outputstring)
fOut.close()


##----- CREATE THE GRAPHS OUTPUT -----##


#Creates the plotting reference.
CSFDandRPLTWindow = plt.figure(1, figsize=(3,8))    #this should be called later when we know the aspect ratio, but for now, I have it here
RSFD = CSFDandRPLTWindow.add_subplot(212)           #http://matplotlib.org/examples/pylab_examples/subplots_demo.html

#To the plot, append the trace.
RSFD.plot(edf_diam, edf_rsfd, color='#D53200', linewidth=1, label="R")
RSFD.fill_between(edf_diam,edf_erro_rsfd_neg,edf_erro_rsfd_pos, alpha=0.2, edgecolor='#D53200', linewidth=0.0, facecolor='#D53200')
RSFD.plot(edf_diam, edf_erro_rsfd_neg_1s, color='#D53200', linewidth=0.25)
RSFD.plot(edf_diam, edf_erro_rsfd_pos_1s, color='#D53200', linewidth=0.25)

#Append the legend to the plot.
RSFD.legend(loc='upper left')

#Set the x-y range to the plot.
RSFD.set_xlim(np.power(10,np.floor(np.log10(min(DIAM_TEMP)))), np.power(10,np.ceil(np.log10(max(DIAM_TEMP)))))
y_min_limit = min(edf_rsfd)
start = find_nearest(edf_diam,min(DIAM_TEMP))
stop  = find_nearest(edf_diam,max(DIAM_TEMP))
y_min = min(edf_rsfd[start:stop])/2.
y_max = edf_erro_rsfd_pos[np.where(edf_rsfd == max(edf_rsfd[start:stop]))]
RSFD.set_ylim(y_min,y_max)

#To the plot, append "+" signs at the bottom that show the actual x-axis locations of the random values
# drawn from that distribution.
RSFD.plot(DIAM_TEMP, np.full(len(DIAM_TEMP),y_min), '|r')

#Make the x-y axes log.
RSFD.set_xscale('log')
RSFD.set_yscale('log')

#Turn on grid lines.
RSFD.grid(b=True, which='minor', color='0.25', linewidth=0.25)
RSFD.grid(b=True, which='major', color='0.25', linewidth=0.25)

#Creates the plotting reference.
CSFD = CSFDandRPLTWindow.add_subplot(211, sharex=RSFD)    #http://matplotlib.org/examples/pylab_examples/subplots_demo.html

#To the plot, append the trace.
CSFD.plot(edf_diam, edf_csfd, color='#D53200', linewidth=1, label="Cumulative")
CSFD.fill_between(edf_diam,edf_erro_csfd_neg,edf_erro_csfd_pos, alpha=0.2, edgecolor='#D53200', linewidth=0.0, facecolor='#D53200')
CSFD.plot(edf_diam, edf_erro_csfd_neg_1s, color='#D53200', linewidth=0.25)
CSFD.plot(edf_diam, edf_erro_csfd_pos_1s, color='#D53200', linewidth=0.25)

#Append the legend to the plot.
CSFD.legend(loc='upper right')

#Set the x-y range to the plot.
CSFD.set_xlim(np.power(10,np.floor(np.log10(min(DIAM_TEMP)))), np.power(10,np.ceil(np.log10(max(DIAM_TEMP)))))
start = find_nearest(edf_diam,min(DIAM_TEMP))
stop  = find_nearest(edf_diam,max(DIAM_TEMP))
y_min = min(edf_csfd[start:stop])/2.
y_max = edf_erro_csfd_pos[np.where(edf_csfd == max(edf_csfd[start:stop]))]
CSFD.set_ylim(y_min,y_max)

#To the plot, append "+" signs at the bottom that show the actual x-axis locations of the random values
# drawn from that distribution.
#CSFD.plot(DIAM_TEMP, np.full(len(DIAM_TEMP),y_min), '|r')

#Make the x-y axes log.
CSFD.set_xscale('log')
CSFD.set_yscale('log')

#Turn on grid lines.
CSFD.grid(b=True, which='minor', color='0.25', linewidth=0.25)
CSFD.grid(b=True, which='major', color='0.25', linewidth=0.25)

#Turn off CSFD bottom axis labels and make it so there is no space between it and the R-plot.
CSFD.tick_params(axis='x', which='both', labelbottom=False)
CSFDandRPLTWindow.subplots_adjust(hspace=0)



#Creates the plotting reference.
DSFDWindow = plt.figure(2, figsize=(3,8))   #this should be called later when we know the aspect ratio, but for now, I have it here
DSFD = DSFDWindow.add_subplot(111)          #http://matplotlib.org/examples/pylab_examples/subplots_demo.html

#To the plot, append the trace.
DSFD.plot(edf_diam, edf_dsfd, color='#D53200', linewidth=1, label="Differential")
DSFD.fill_between(edf_diam,edf_erro_dsfd_neg,edf_erro_dsfd_pos, alpha=0.2, edgecolor='#D53200', linewidth=0.0, facecolor='#D53200')
DSFD.plot(edf_diam, edf_erro_dsfd_neg_1s, color='#D53200', linewidth=0.25)
DSFD.plot(edf_diam, edf_erro_dsfd_pos_1s, color='#D53200', linewidth=0.25)

#Append the legend to the plot.
DSFD.legend(loc='upper right')

#Set the x-y range to the plot.
DSFD.set_xlim(np.power(10,np.floor(np.log10(min(DIAM_TEMP)))), np.power(10,np.ceil(np.log10(max(DIAM_TEMP)))))
start = find_nearest(edf_diam,min(DIAM_TEMP))
stop  = find_nearest(edf_diam,max(DIAM_TEMP))
y_min = min(edf_dsfd[start:stop])
y_max = edf_erro_dsfd_pos[np.where(edf_dsfd == max(edf_dsfd[start:stop]))]
DSFD.set_ylim(y_min,y_max)

#To the plot, append "+" signs at the bottom that show the actual x-axis locations of the random values
# drawn from that distribution.
DSFD.plot(DIAM_TEMP, np.full(len(DIAM_TEMP),y_min), '|r')

#Make the x-y axes log.
DSFD.set_xscale('log')
DSFD.set_yscale('log')

#Turn on grid lines.
DSFD.grid(b=True, which='minor', color='0.25', linewidth=0.25)
DSFD.grid(b=True, which='major', color='0.25', linewidth=0.25)



#Creates the plotting reference.
ISFDWindow = plt.figure(3, figsize=(3,8))   #this should be called later when we know the aspect ratio, but for now, I have it here
ISFD = ISFDWindow.add_subplot(111)          #http://matplotlib.org/examples/pylab_examples/subplots_demo.html

#To the plot, append the trace.
ISFD.plot(edf_diam, edf_isfd, color='#D53200', linewidth=1, label="Incremental")
ISFD.fill_between(edf_diam,edf_erro_isfd_neg,edf_erro_isfd_pos, alpha=0.2, edgecolor='#D53200', linewidth=0.0, facecolor='#D53200')
ISFD.plot(edf_diam, edf_erro_isfd_neg_1s, color='#D53200', linewidth=0.25)
ISFD.plot(edf_diam, edf_erro_isfd_pos_1s, color='#D53200', linewidth=0.25)

#Append the legend to the plot.
ISFD.legend(loc='upper right')

#Set the x-y range to the plot.
ISFD.set_xlim(np.power(10,np.floor(np.log10(min(DIAM_TEMP)))), np.power(10,np.ceil(np.log10(max(DIAM_TEMP)))))
start = find_nearest(edf_diam,min(DIAM_TEMP))
stop  = find_nearest(edf_diam,max(DIAM_TEMP))
y_min = min(edf_isfd[start:stop])
y_max = edf_erro_isfd_pos[np.where(edf_isfd == max(edf_isfd[start:stop]))]
ISFD.set_ylim(y_min,y_max)

#To the plot, append "+" signs at the bottom that show the actual x-axis locations of the random values
# drawn from that distribution.
ISFD.plot(DIAM_TEMP, np.full(len(DIAM_TEMP),y_min), '|r')

#Make the x-y axes log.
ISFD.set_xscale('log')
ISFD.set_yscale('log')

#Turn on grid lines.
ISFD.grid(b=True, which='minor', color='0.25', linewidth=0.25)
ISFD.grid(b=True, which='major', color='0.25', linewidth=0.25)

#And, show the plots.
plt.show()
