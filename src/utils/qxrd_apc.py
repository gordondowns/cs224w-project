# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 09:23:08 2018

@author: Gordon Downs
"""

# weights of 2/3, 1/3, and startswith(num)
# Here is Cu: X-RAY WAVELENGTHS:   1.540562  1.544390 (1.5)
# Mo: X-RAY WAVELENGTHS:   0.709300  0.713590 (0.7)
# Co: X-RAY WAVELENGTHS:   1.788965  1.792850 (1.7)


import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import norm
from scipy.optimize import minimize#, fmin_l_bfgs_b
from time import time
from collections import deque
# import numba
#from CifFile import ReadCif

#import warnings
#warnings.filterwarnings("error")


class XrayPhase(object):
    """The x-ray diffraction data of one material. Can be either a Profile (full pattern XY data)
    or a Dif (peaks only, which will be broadened). Has scalings, RIR, chemistry, and name of corresponding file."""
    
    def __init__(self,file_path=None,file_nickname=None):
        self.RIR = None
        self.scaling = None
        self.scaling_bounds = (0.0,None)
        self.refine_scaling = True
        self.chemistry = None
        self.relative_contribution = None
        self.absolute_contribution = None
        self.file_path = file_path
        if (file_nickname is None) and (file_path is not None):
            self.file_nickname = file_path.split('/')[-1].split('\\')[-1].split('.')[0]
        else:
            self.file_nickname = file_nickname
    
    def SetScaling(self,scaling):
        self.scaling = scaling

    def SetScalingBounds(self,scaling_bounds):
        self.scaling_bounds = scaling_bounds
        
    def RefineScaling(self,boolean):
        self.refine_scaling = boolean

    def GetArea(self,start_2theta=None,end_2theta=None):
        pass
    
    def GetAreaSloppy(self,start_2theta,end_2theta):
        pass
    
    def ReadFile(self,file_path): 
        pass

    def SetRelativeContribution(self,percentage):
        self.relative_contribution = percentage
    def GetRelativeContribution(self):
        return self.relative_contribution
    def SetAbsoluteContribution(self,percentage):
        self.absolute_contribution = percentage
    def GetAbsoluteContribution(self):
        return self.absolute_contribution



class XrayProfile(XrayPhase):
    """The x-ray diffraction data of one material, with full pattern XY data. Can be read in from MDI, XY, or CSV."""
    
    def __init__(self, f, file_path, is_input_profile=False, file_nickname=None, take_every_nth_point=1, twotheta_ranges=[(0.0,90.0)], twotheta_offset=0.0, print_warnings=True, normalize=True, xy=None, xy_scaled=None, RIR=0.0):
        super().__init__(file_path,file_nickname)
        self.xy_data = None
        self.xy_data_for_plotting = None
        self.x_data_interpolated = None
        self.y_data_interpolated = None
        self.area_with_scaling_1 = None
        if(is_input_profile):
            self.SetScaling(1.0)
        if f:
            self.ReadFile(f,print_warnings=print_warnings,is_input_profile=is_input_profile)
        else:
            self.xy_data_unprocessed = xy
            self.RIR = RIR
            self.chemistry = {}
        self.process_xy(is_input_profile=is_input_profile, take_every_nth_point=take_every_nth_point, twotheta_ranges=twotheta_ranges, twotheta_offset=twotheta_offset, normalize=normalize)
    
    def GetInterpolatedXYData(self, xValues):
        #print "GetInterpolatedXYData.xValues",xValues
        
        self.x_data_interpolated = np.array(xValues)
        self.y_data_interpolated = np.interp(xValues,self.xy_data[0],self.xy_data[1])
        #FIXME: implement this
        return np.array([self.x_data_interpolated,self.y_data_interpolated])
    
    def GetScaledPattern(self,xValues,scaling=None):
        if(scaling is None):
            scaling = self.scaling
        #if(not xValues is None): #TODO: implement this
        self.GetInterpolatedXYData(xValues)
        return np.multiply( self.y_data_interpolated, scaling )
    
    def GetArea(self, xValues= None, scaling=None, start_2theta=None, end_2theta=None):
        from scipy import integrate
        if scaling is None:
            scaling = self.scaling
        if xValues is not None:
            self.GetInterpolatedXYData(xValues)
            self.area_with_scaling_1 = integrate.simps(self.y_data_interpolated, xValues)
        
        return self.area_with_scaling_1 * scaling
        #FIXME: Implement ability to set start and end 2thetas to None
    
    # profile
    def ReadFile(self, f, print_warnings=True, is_input_profile=False):
        
        RIR_found, chemistry_found = False, False
        RIR = 0.0
        chemistry = {}
        should_read_next_line = True
        mdi_header_found, xy_data_found = False, False
        # print(f.readlines())
        while(True):
            if should_read_next_line:
                line = f.readline()
                # print('1',line)
            else:
                should_read_next_line = True
            if line == "":
                break
            elif "".join(line.split()).startswith("#RIR"):
                RIR_found = True
                RIR = float(line.lstrip().split()[2])
            elif "".join(line.split()).startswith("#CHEMISTRY"):
                chemistry_found = True
                while(True):
                    line = f.readline()
                    should_read_next_line = False
                    linesplit = line.split()
                    if len(linesplit) != 2 or not linesplit[1].replace('.','').isdigit():
                        # chemistry not found on this line
                        break
                    else:
                        # chemistry found on this line
                        name,percentage = linesplit
                        chemistry[name] = float(percentage)
            else:
                try: # look for MDI format
                    linesplit = line.replace('.','').split()
#                    if line.startswith('2.0 0.05'):
#                        pass
#                        print linesplit
                    if (len(linesplit) == 7 and linesplit[0].isdigit() and linesplit[1].isdigit() and
                    linesplit[5].isdigit() and linesplit[6].isdigit()):
                        
                        mdi_header_found = True
                        linesplit = line.split()
                        x_start = float(linesplit[0])
                        #x_increment = float(line2split[1])
                        #wavelength = float(linesplit[4])
                        x_end = float(linesplit[5])
                        x_num = int(linesplit[6])
                        
                        x_data = np.linspace(x_start,x_end,num=x_num)
                        y_data = []
                        
                        while(True):
                            line = f.readline()
                            # print('2',line)
                            should_read_next_line = False
                            linesplit = line.replace('.','').split()
                            if "".join(linesplit).isdigit():
                                y_data += [float(i) for i in line.split()]
                            else:
                                break
                except:
                    pass
                try: # look for XY or CSV format
                    linesplit = line.replace(',',' ').replace('.','').replace('e','').replace('E','').replace('-','').replace('+','').split()
                    
                    if len(linesplit) == 2 and linesplit[0].isdigit() and linesplit[1].isdigit():
                        linesplit = line.replace(',',' ').split()
                        x_data = [float(linesplit[0])]
                        y_data = [float(linesplit[1])]
                        xy_data_found = True
                        
                        while(True):
                            line = f.readline()
                            # print('3',line)
                            should_read_next_line = False
                            linesplit = line.replace(',',' ').replace('.','').replace('e','').replace('E','').replace('-','').replace('+','').split()
                            if len(linesplit) == 2 and linesplit[0].isdigit() and linesplit[1].isdigit():
                                linesplit = line.replace(',',' ').split()
                                x_data += [float(linesplit[0])]
                                y_data += [float(linesplit[1])]
                            else:
                                break
                except:
                    pass
        if not (mdi_header_found and len(x_data) == len(y_data) or xy_data_found):
            # f.close()
            if self is None:
                fp = '<file path not given>'
            else:
                fp = self.file_path
            raise SyntaxError("No data found in profile: '"+fp+"'\n"
                             +'Profile files must contain either XY data or MDI data.\n'
                             +'- XY files must contain at least one line that is\n'
                             +'  a pair of numbers separated by whitespace and/or a comma.\n'
                             +'- MDI files must contain standard MDI header line of the form:\n'
                             +'  x_start x_increment scan_rate xray_source xray_wavelength x_end number_of_points')
        
        # remove leading and trailing 0s
        i_first, i_last = 0, len(y_data)-1
        while y_data[i_first] == 0.0:
            i_first += 1
        while y_data[i_last] == 0.0:
            i_last -= 1
        if i_first < i_last: # make sure not to throw an exception if the pattern is all 0s
            x_data = x_data[i_first:i_last+1]
            y_data = y_data[i_first:i_last+1]
        
        # flip the ordering if the data aren't in ascending order
        if x_data[0] > x_data[-1]:
            x_data = x_data[::-1]
            y_data = y_data[::-1]
        
        # At this point, at least x_data and y_data have been found, and RIR and chemistry have been looked for.
        if self is None:
            if type(x_data) is np.ndarray:
                return x_data.tolist(),y_data
            else:
                return x_data,y_data

        self.RIR = RIR
        self.chemistry = chemistry
        if print_warnings:
            if(not is_input_profile and not RIR_found):
                print("Warning: no RIR found in profile: "+self.file_path)
            if(not chemistry_found):
                print("Warning: no chemistry found in profile: "+self.file_path)

        self.xy_data_unprocessed = np.array([x_data,y_data])
        return self.xy_data_unprocessed
        
    def process_xy(self, is_input_profile=False, take_every_nth_point=1, twotheta_ranges=[(0.0,90.0)], twotheta_offset=0.0, normalize=True):
        x_data = self.xy_data_unprocessed[0,:]
        y_data = self.xy_data_unprocessed[1,:]
        
        # apply 2theta offset: add offset to every 2theta value
        if twotheta_offset != 0.0:
            x_data = [x + twotheta_offset for x in x_data]
        
        # set minimum intensity to 0.0 (if not input profile) and set maximum intensity to 100.0
        if normalize:
            maxY = max(y_data)
            if is_input_profile:
                y_data = np.array([y_data[i]*100.0/maxY for i in range(len(y_data))])
            else:
                minY = min(y_data)
                y_data = np.array([(y_data[i]-minY) *100.0/(maxY-minY) for i in range(len(y_data))])
        
        self.xy_data_for_plotting = np.array([x_data,y_data])

        if take_every_nth_point > 1:
            x_data = [x_data[i] for i in range(len(x_data)) if i % take_every_nth_point == 0]
            y_data = [y_data[i] for i in range(len(y_data)) if i % take_every_nth_point == 0]
        
        # remove data points outside of desired range
        if twotheta_ranges != [(0.0,90.0)]:
            new_x_data,new_y_data = [],[]
            for x,y in zip(x_data,y_data):
                for xmin,xmax in twotheta_ranges:
                    if xmin <= x <= xmax:
                        new_x_data.append(x)
                        new_y_data.append(y)
                        break # It is sufficient for the data to be in only one interval. So, go to next x_data.
            x_data,y_data = new_x_data,new_y_data
        
        self.xy_data = np.array([x_data,y_data])
        

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################



class TopLevel(object):
    '''Top level class that wraps everything together. This the only thing the end user interacts with.'''
    
    def __init__(self, input_file_path, x=None,y=None, twotheta_ranges=[(0.0,90.0)], take_every_nth_point=1, print_warnings = True, cell_parameter_type_for_optimization = 'reciprocal', background_fitting_method=None):
        
        self.Set2ThetaRanges(twotheta_ranges,apply_to_profiles=False)
        self.twotheta_offset = 0.0
        self.refine_twotheta_offset = False
        self.twotheta_offset_bounds = (None,None)
        
        self.print_warnings = print_warnings
        self.take_every_nth_point = take_every_nth_point
        
        if x is None: # file initialization
            with open(input_file_path, 'r') as f:
                self.input_profile = XrayProfile(f,input_file_path,is_input_profile=True, take_every_nth_point=take_every_nth_point, twotheta_ranges=twotheta_ranges, print_warnings=self.print_warnings)
        else: # ODR initialization
            xy = np.array([x,y])
            p = XrayProfile(f=None,
                            xy=xy,
                            # xy_scaled=xy_scaled,
                            RIR=0,
                            file_nickname=input_file_path,
                            file_path=input_file_path,
                            twotheta_offset=0.0,
                            twotheta_ranges=twotheta_ranges,
                            print_warnings=self.print_warnings,
                            take_every_nth_point=take_every_nth_point,
                            is_input_profile=True)
            self.input_profile = p
        
        self.background_fitting_method = background_fitting_method
        if background_fitting_method:
            raise Exception('no background fitting method available. import xrd_background_removal.')
            # _,bg = self.input_profile.StripPeaks(keep="none")
            x,y = self.input_profile.xy_data
            _,bg = xrd_background_removal.removebg(x,y,method=background_fitting_method)
            self.background_profiles = [np.ones_like(bg),bg]
            self.background_scalings = [0.0,1.0]
            self.background_bounds = [(None,None),(None,None)]
            self.refine_background = [True,False]
        else:
            self.background_profiles = [np.ones_like(self.input_profile.xy_data[0])]
            self.background_scalings = [min(self.input_profile.xy_data[1])]
            self.background_bounds = [(None,None)]
            self.refine_background = [True]
        
        self.difs = []
        self.profiles = []
        self.True_for_area_percent_False_for_wt_percent = False
        self.algorithm = "2-step"
        self.sol = None
        self.cell_parameter_type_for_optimization = cell_parameter_type_for_optimization
        
        self.all_oxide_names = None
        self.calculated_oxide_percentages = None
        
        self.algorithm_run_time = None
        self.algorithm_number_of_iterations = None
        self.norm_between_optimized_profile_and_input = None
    
    def AddProfile(self,file_path,scaling=0.1,scaling_bounds=(0.0,None),refine_scaling=True,twotheta_offset=0.0,normalize=True):
        with open(file_path, 'r') as f:
            p = XrayProfile(f,file_path,is_input_profile=False,twotheta_offset=0.0,print_warnings=self.print_warnings,normalize=normalize)
        p.SetScaling(scaling)
        p.SetScalingBounds(scaling_bounds)
        p.refine_scaling = refine_scaling
        p.twotheta_offset = twotheta_offset
        self.profiles.append(p)
    
    def SetBackground(self, background_scalings):
        if background_scalings.__class__ in (float,int):
            background_scalings = [background_scalings]
        self.background_scalings = background_scalings
        self.background_bounds = [(None,None)] * len(background_scalings)
    
    def SetBackgroundBounds(self,background_bounds):
        self.background_bounds = background_bounds
    
    def RefineBackground(self, boolean_list):
        if boolean_list.__class__ in (float,int):
            boolean_list = [boolean_list]
        self.refine_background = boolean_list
        
    def Set2ThetaOffset(self, offset):
        self.twotheta_offset = offset
    
    def Set2ThetaOffsetBounds(self,offset_bounds):
        self.twotheta_offset_bounds = offset_bounds

    def Refine2ThetaOffset(self,boolean):
        self.refine_twotheta_offset = boolean
    
    def Set2ThetaRanges(self, list_of_ranges,apply_to_profiles=True):
        # Can input either like [start,end] or [[start1,end1],(start2,end2), ...]
        # but will always be stored like [(start1,end1),(start2,end2), ...]
        if list_of_ranges is None:
            self.twotheta_ranges = [(0.0,90.0)]
        if isinstance(list_of_ranges[0], (int, float)):
            # list_of_ranges was input like [start,end]
            self.twotheta_ranges = [(float(list_of_ranges[0]),float(list_of_ranges[1]))]
        else:
            # list_of_ranges was input like [[start1,end1],...]
            self.twotheta_ranges = []
            for pair in list_of_ranges:
                self.twotheta_ranges += [(float(pair[0]),float(pair[1]))]
        
        if apply_to_profiles:
            # apply new ranges to background profiles
            prior_x = self.input_profile.xy_data[0]
            self.background_profiles = [p[np.logical_or.reduce([np.logical_and(lb<=prior_x,prior_x<=ub) for (lb,ub) in self.twotheta_ranges])]
                                        for p in self.background_profiles]
            # apply new ranges to input profile
            self.input_profile.process_xy(is_input_profile=True, 
                                        take_every_nth_point=self.take_every_nth_point,
                                        twotheta_ranges=self.twotheta_ranges,
                                        twotheta_offset=self.twotheta_offset,
                                        normalize=True)
            # apply new ranges to added profiles
            for p in self.profiles:
                p.process_xy(is_input_profile=False, 
                            take_every_nth_point=self.take_every_nth_point,
                            twotheta_ranges=self.twotheta_ranges,
                            twotheta_offset=self.twotheta_offset,
                            normalize=True)
            
    def SumDifsAndProfiles(self,true_for_chopped_data_false_for_full_original=True):

        if true_for_chopped_data_false_for_full_original:
            x_values = self.input_profile.xy_data[0]
        else:
            x_values = self.input_profile.xy_data_for_plotting[0]
        
        y_out = np.dot(self.background_scalings, self.background_profiles)
        
        for d in self.difs:
            y_out += d.GetBroadenedPattern(x_values)

        for p in self.profiles:
            y_out += p.GetScaledPattern(x_values)

        return y_out

    def InitializeBackgroundToMinimumOfInputPattern(self, degree = 0):
        inputXY = self.input_profile.xy_data
        self.background_profiles = [np.ones_like(inputXY[0])]
        for degree in range(1,degree+1):
            self.background_profiles.append( np.power(inputXY[0],degree) )
        self.background_scalings = [min(inputXY[1])] + [0.0]*(degree)
        self.background_bounds = [(None,None)]*(degree+1)
        self.refine_background = [True]*(degree+1)
    
    def GetNorm(self):
        y_out = self.SumDifsAndProfiles()
        self.norm_between_optimized_profile_and_input = np.linalg.norm(self.input_profile.xy_data[1] - y_out)
        return self.norm_between_optimized_profile_and_input
    
    def DoPlot(self,image_file_path=None,figsize=None,range_to_plot=None,linewidth=0.75,title=None,
            plot_optimized_profile_and_difference=True,plot_input_pattern_and_background=True):
        
        self.CalculateChemistry()
        
        difs = self.difs
        profiles = self.profiles
        
        if figsize is None and image_file_path is not None:
            figsize=(30,15)
        my_plot = plt.figure(1,figsize=figsize)
        if title:
            plt.title(title)
        
        inputXY = np.array(self.input_profile.xy_data_for_plotting)
        if self.twotheta_offset != 0.0:
            inputXY[0] = [inputXY[0][i] + self.twotheta_offset for i in range(len(inputXY[0]))]
            for p in self.profiles:
                p.GetInterpolatedXYData(inputXY[0])
        
        if range_to_plot is None:   #TODO: serious bug: need to change background implementation so that I store "full background for plotting"
                                    # as well as the "x-trimmed background for optimizing" (current implementation)
            if self.twotheta_ranges == [(0.0,90.0)]:
                x_min, x_max = min(inputXY[0]), max(inputXY[0])
                global_multiplier = 100.0/max(inputXY[1])
            else:
                x_min = min([pair[0] for pair in self.twotheta_ranges])
                x_max = max([pair[1] for pair in self.twotheta_ranges])
                global_multiplier = 100.0/max([inputXY[1][i] for i in range(len(inputXY[0])) if (inputXY[0][i] >= x_min and inputXY[0][i] <= x_max)])
        else:
            x_min, x_max = range_to_plot[0], range_to_plot[1]
            global_multiplier = 100.0/max([inputXY[1][i] for i in range(len(inputXY[0])) if (inputXY[0][i] >= x_min and inputXY[0][i] <= x_max)])
        
        optimizedY = self.SumDifsAndProfiles(true_for_chopped_data_false_for_full_original=False)
        
        background = np.dot(self.background_scalings, self.background_profiles)
        
        for p in profiles:
            if(p.RIR < 0.001):
                plt.plot(inputXY[0],(p.GetScaledPattern(inputXY[0])+background)*global_multiplier,label=p.file_nickname,linewidth=linewidth)
            else:
                if(self.True_for_area_percent_False_for_wt_percent):
                    plt.plot(inputXY[0],(p.GetScaledPattern(inputXY[0])+background)*global_multiplier,label=p.file_nickname+" "+
                             "{0:.1f}".format(p.GetRelativeContribution())+"%",linewidth=linewidth)
                else:
#                    plt.plot(inputXY[0],(p.GetScaledPattern(inputXY[0])+background)*global_multiplier,label=p.file_nickname+" "+
#                             "{0:.1f}".format(p.GetRelativeContribution())+" wt%",linewidth=linewidth)
                    plt.plot(inputXY[0],(p.GetScaledPattern(inputXY[0])+background)*global_multiplier,label=p.file_nickname,linewidth=linewidth)
    
        for d in difs:
            if(d.RIR < 0.001):
                plt.plot(inputXY[0],(d.GetBroadenedPattern(inputXY[0])+background)*global_multiplier,label=d.file_nickname,linewidth=linewidth)
            else:
                if(self.True_for_area_percent_False_for_wt_percent):
                    plt.plot(inputXY[0],(d.GetBroadenedPattern(inputXY[0])+background)*global_multiplier,label=d.file_nickname+" "+
                        "{0:.1f}".format(d.GetRelativeContribution())+"%",linewidth=linewidth)
                else:
#                    plt.plot(inputXY[0],d.GetBroadenedPattern(inputXY[0])+background,label=d.file_nickname+" "+
#                        "{0:.2f}".format(d.GetRelativeContribution())+" wt%",linewidth=linewidth)
                    plt.plot(inputXY[0],(d.GetBroadenedPattern(inputXY[0])+background)*global_multiplier,label=d.file_nickname,linewidth=linewidth)
        if plot_input_pattern_and_background:
            plt.plot(inputXY[0],background*global_multiplier,label="background",linestyle="dotted",color="red",linewidth=linewidth)
            plt.plot(inputXY[0],inputXY[1]*global_multiplier,label="input:"+self.input_profile.file_nickname,color="black",linewidth=linewidth)
        if plot_optimized_profile_and_difference:
            plt.plot(inputXY[0],optimizedY*global_multiplier,label="optimized profile",color="magenta",linewidth=linewidth)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.0, -0.05))
        plt.xlabel('2-theta (deg)')
        plt.ylabel('intensity')
        if plot_optimized_profile_and_difference:
            difference = (inputXY[1] - optimizedY)*global_multiplier
            offsetline = np.full_like(inputXY[0],110.0)
            plt.plot(inputXY[0], difference+offsetline, linestyle="solid", color="pink",linewidth=linewidth)
            plt.plot(inputXY[0], offsetline, linestyle="dotted", color="red")
            plt.axis([x_min, x_max, 0.0 ,120.0])
        else:
            plt.axis([x_min, x_max, 0.0 ,105.0])
        if image_file_path is None:
            # plt.tight_layout()
            plt.show()
        else:
            plt.savefig(image_file_path, bbox_inches='tight')  
            plt.close()
    