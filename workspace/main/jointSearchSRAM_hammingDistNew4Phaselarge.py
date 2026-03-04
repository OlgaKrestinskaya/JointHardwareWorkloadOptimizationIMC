from _import_scripts import scripts
from scripts.notebook_utils import *
import csv
from scripts import *
import time

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, FloatRandomSampling
from pymoo.optimize import minimize
import math
import json
from scripts_new import *
import os
import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.problem import ElementwiseProblem

from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
import random
# Directory you want to create
EXTRA_COMPONENT = """
- !Component # Column readout (ADC)
  name: {}
  <<<: [*component_defaults, *keep_outputs, *no_coalesce]
  subclass: dummy_storage
  attributes: {{width: ENCODED_OUTPUT_BITS, n_bits: width, <<<: *cim_component_attributes}}
"""

# NeuroSim has some extra digital components. For fair comparison, we'll add as
# many components it has. We're not using them for energy so we'll just realize
# all of them as intadders.
EXTRA_NEUROSIM_COMPONENTS = ["shift_add", "adder", "pooling", "activation"]
EXTRA_COMPONENTS_CONFIG = "\n".join(
    EXTRA_COMPONENT.format(name) for name in EXTRA_NEUROSIM_COMPONENTS
)
countl=0
def run_layer(
    dnn: str,
    layer: str,
    #avg_input: float,
    #avg_weight: float,
    #shape: tuple,
    params: list,
    max_mappings: int = None,
):

    
    vl,bl,csX,csY,srg,tic,mit,gbp=params
    bps=1
    vl,bl =[float(x) for x in [vl,bl]]
    bps,csX,csY,srg,tic,mit,gbp=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
    #print(vl,bps,bl,csX,csY,srg,tic,mit,gbp)
    mis=1
    spec = get_spec(
        "basic_analog_sram",
        #tile="input_output_bufs",
        tile="input_output_bufs",
        #chip="large_router",
        chip="large_router_glb",
        #system="ws_chip2chip_link",
        system="fetch_weights_lpddr4",
        #system="ws_dummy_buffer_many_macro",
        dnn=dnn,  # Set the DNN and layer
        layer=layer,
        jinja_parse_data={
            "cell_override": "sram_neurosim_default.cell.yaml",
            "ignoreme_placeholder": EXTRA_COMPONENTS_CONFIG,
        },
    )

    # NeuroSim's default macro variable settings
    spec.variables.update(
        dict(
            #INPUT_ENCODING_FUNC="offset_encode_if_signed_hist", #BASIC ANALOG ALREADY has it's own encoding
            #WEIGHT_ENCODING_FUNC="offset_encode_if_signed_hist",
            VOLTAGE=vl, #can cahnge this (?) was 0.85
            TECHNOLOGY=32,  # nm  --> adjust this
            BITS_PER_CELL=bps, # vary this was 2
            ADC_RESOLUTION=5, #vary this
            VOLTAGE_DAC_RESOLUTION=1,
            TEMPORAL_DAC_RESOLUTION=1,
            N_SHIFT_ADDS_PER_BANK=16, #not used on the code (?)
            N_ADC_PER_BANK=16, #vary this
            BASE_LATENCY=bl,  # For near-zero leakage, make it really fast. was 1e-12
            READ_PULSE_WIDTH=1e-8,
            VOLTAGE_ENERGY_SCALE=1,
            VOLTAGE_LATENCY_SCALE=1,
            #AVERAGE_INPUT_VALUE=float(avg_input), #in common variables
            #AVERAGE_WEIGHT_VALUE=float(avg_weight),
            BATCH_SIZE=1,
        )
    )
    spec.architecture.find("row").spatial.meshY = csX  #vary this
    spec.architecture.find("column").spatial.meshX = csY #vary this
    spec.architecture.find("adc").attributes[
        "adc_estimator_plug_in"
    ] = '"Neurosim Plug-In"'

    #variables to vary:
    #spec.variables["N_ADC_PER_BANK"] = 8 #varing number of ADCs per crossbar --> so far nothign changes because of this!!! need to check MACRO separately!!!!
    # Set the shape of the layer. NeuroSim uses a different shape than Timeloop
    
    #spec.variables["BITS_PER_CELL"] = 2
    #spec.variables["CIM_UNIT_WIDTH_CELLS"] = 1
    #spec.variables["VOLTAGE"] = 0.85
    #spec.variables["BASE_LATENCY"] = 1e-12
    

    #variables dependent on the software parames:
    input_bits=8
    weights_bits=8
    output_bits=8
    spec.variables["WEIGHT_BITS"] = weights_bits
    spec.variables["INPUT_BITS"] = input_bits
    spec.variables["OUTPUT_BITS"] = output_bits
    spec.variables["TEMPORAL_DAC_RESOLUTION"] = input_bits
    #spec.variables["ADC_RESOLUTION"] = 8 #this can be varied up to certain extent
    
    # Enable the MAX_UTILIZATION variable. This will generate a
    # workload that maximizes the utilization of the array.
    spec.variables["MAX_UTILIZATION"] = False  # Do NOT generate a maximum-utilization workload; we're running a DNN workload.
    #ins = spec.problem.instance
    #ins["P"] = (shape[0] - shape[3] + 1) // shape[7]
    #ins["Q"] = (shape[1] - shape[4] + 1) // shape[7]
    #ins["C"] = shape[2]
    #ins["R"] = shape[3]
    #ins["S"] = shape[3]
    #ins["M"] = shape[5]
    #ins["WStride"] = shape[7]
    #ins["HStride"] = shape[7]

    # Lock in the mapping to only evaluate one mapping by defualt
    #dt = spec.architecture.find("dummy_top").constraints.temporal
    #dt = spec.architecture.find("chip2chip_link").constraints.temporal
    #dt.factors_only = dt.factors
    #dt.factors.clear()
    #dt.factors_only.add_eq_factor("X", input_bits)
    #dt.factors_only.add_eq_factor("P", -1)
    #dt.factors_only.add_eq_factor("Q", -1)
    #spec.mapping.max_permutations_per_if_visit = 1

    #dt = spec.architecture.find("glb").constraints.temporal
    #dt.factors_only = dt.factors
    #dt.factors.clear()
    #dt.factors_only.add_eq_factor("X", input_bits)
    #dt.factors_only.add_eq_factor("P", -1)
    #dt.factors_only.add_eq_factor("Q", -1)
    #spec.mapping.max_permutations_per_if_visit = 1

    spec.architecture.find("chip_in_system").spatial.meshX = 1 #300 works with simple case for ResNet18
    spec.architecture.find("chip_in_system").attributes.has_power_gating = True
    spec.architecture.find("chip_in_system").constraints.spatial.no_reuse = []
    
    #SYSTEM LVL
    #spec.architecture.find("macro_in_system").spatial.meshX = mis #300 works with simple case for ResNet18
    #spec.architecture.find("macro_in_system").attributes.has_power_gating = True
    #spec.architecture.find("chip_in_system").spatial.meshX = 300 #300 works with simple case for ResNet18
    #spec.architecture.find("chip_in_system").attributes.has_power_gating = True
    #spec.architecture.find("macro_in_system").constraints.spatial.no_reuse = []
    
    # ARCHITECTURE LVL
    spec.architecture.find("shared_router_group").spatial.meshX = srg
    spec.architecture.find("glb").attributes.depth = gbp #int(1024*1024*16*8/2048) 
    #spec.architecture.find("router").attributes.width = 32 #not sure of this effects anything???
    spec.architecture.find("shared_router_group").attributes.has_power_gating = True
    #spec.architecture.find("shared_router_group").constraints.spatial.no_reuse = []
    spec.architecture.find("tile_in_chip").spatial.meshX = tic #2
    #spec.architecture.find("tile_in_chip").spatial.meshY = 2 doesn't work!
    spec.architecture.find("tile_in_chip").attributes.has_power_gating = True
    #spec.architecture.find("tile_in_chip").constraints.spatial.no_reuse = []

    #TILE LVL
    spec.architecture.find("macro_in_tile").spatial.meshX = mit #8
    #spec.architecture.find("macro_in_tile").spatial.meshY = 8 (meshY is not working with mapping?)
    spec.architecture.find("macro_in_tile").attributes.has_power_gating = True
    #spec.architecture.find("macro_in_tile").constraints.spatial.no_reuse = []

    # If there's a max_mappings, expand the search space
    if max_mappings is not None:
        # Set to evaluate max_mappings mappings
        spec.mapper.search_size = max_mappings
        spec.mapper.max_permutations_per_if_visit = max_mappings
        spec.mapper.victory_condition = max_mappings

        # Expand the problem and relax constraints to expand the search space
        #for d in "RSMCPQXYZG":
        #    print(d)
        #    print(spec.problem.instance[d])
        #    spec.problem.instance[d] = 8
        #    print(spec.problem.instance[d])
        # change this accordingly!!! --> ADD SYSTEM PART
        for t in ["row", "column", "macro", "dummy_top", "1bit_x_1bit_mac", "input_buffer", "output_buffer", "macro_in_tile", "glb", "router"]:
            t = spec.architecture.find(t)
            t.constraints.spatial.permutation.clear()
            t.constraints.temporal.permutation.clear()

    # Run the mapper
    return run_mapper(spec)


def getSpecs(dnn_name,layer_name):
    spec = get_spec(
        "basic_analog_sram",
        #tile="input_output_bufs",
        tile="input_output_bufs",
        #chip="large_router",
        chip="large_router_glb",
        #system="ws_chip2chip_link",
        #system="ws_dummy_buffer_many_macro",
        system="fetch_weights_lpddr4",
        dnn=dnn_name,  # Set the DNN and layer
        layer=layer_name,
        jinja_parse_data={
            "cell_override": "sram_neurosim_default.cell.yaml",
            "ignoreme_placeholder": EXTRA_COMPONENTS_CONFIG,
        },
    )
    return spec

def divide_list_unequal(input_list, sizes):
    """
    Divides a list into unequal parts based on the given sizes.
    
    Parameters:
        input_list (list): The original list to be divided.
        sizes (list): A list of integers specifying the size of each part.
        
    Returns:
        list of lists: A list containing the divided parts.
    """
    if sum(sizes) > len(input_list):
        raise ValueError("The sum of sizes exceeds the length of the input list.")

    result = []
    start_index = 0
    
    for size in sizes:
        part = input_list[start_index:start_index + size]
        result.append(part)
        start_index += size
        
    return result

def getMetricsX(paramsLST,dnn_names):
    mydict={}
    for dnn_name in dnn_names:
    #    DNN = dnn_name
    #    layers = [f for f in os.listdir(f"../models/workloads/{DNN}") if f != "index.yaml" and f.endswith(".yaml")]
    #    layers = sorted(l.split(".")[0] for l in layers)
    
        # # CiMLoop One Mapping
        #start_time_one_mapping = time.time()
     #   results = joblib.Parallel(n_jobs=64)(
     #       joblib.delayed(run_layer)(DNN, layer, paramsLST,1000)
      #      for layer in layers
            #joblib.delayed(run_layer)(DNN, layer, avg_input, avg_weight, shape,paramsLST,1000)
            #for layer, avg_input, avg_weight, shape in zip(
            #    layers, input_averages, weight_averages, SHAPES
            #)
     #   )
        #end_time_one_mapping = time.time()
    
        #print('time per one:',end_time_one_mapping-start_time_one_mapping)
      
                
        #avg_sum_tops_per_mm2=sum_tops_per_mm2/len(results)        
        #avg_sum_tops_per_w=sum_tops_per_w/len(results)
        #avg_computes_per_second=sum_computes_per_second/len(results)
        #avg_computes_per_second_per_square_meter=sum_computes_per_second_per_square_meter/len(results)
        mydict[dnn_name]=[0, 0, 0, 0, 0, 0, 0, 0]
    
        #print('total_energy in J:',total_energy)
        #print('total_area in cm2:',total_area)
        #print('total_latency in s:',total_latency)
        #print('total_mac:',total_mac)
        #print('avg_sum_tops_per_mm2:',avg_sum_tops_per_mm2)
        #print('avg_sum_tops_per_w:',avg_sum_tops_per_w)
        #print('avg_computes_per_second',avg_computes_per_second)
        #print('avg_computes_per_second_per_square_meter',avg_computes_per_second_per_square_meter)
        #print('max_percent_utilization:',max_percent_utilization)
        #print('area_utilized in cm2:',max_percent_utilization*total_area)
        
        #print(results[0].mapping)
    return mydict

def getMetricsParallel(paramsLST,dnn_names,num_mapping,n_jobs):


    mydict={}
    layers_allnets={}
    networks_sizes=[]
    for dnn_name in dnn_names:
        DNN = dnn_name
        layers = [f for f in os.listdir(f"../models/workloads/{DNN}") if f != "index.yaml" and f.endswith(".yaml")]
        layers = sorted(l.split(".")[0] for l in layers)
        #print(layers)
        layers_allnets[dnn_name]=layers
        networks_sizes.append(len(layers))
    #print(layers_allnets["resnet18"])
    
    #for dnn_name in dnn_names:
        #DNN = dnn_name
        #layers = [f for f in os.listdir(f"../models/workloads/{DNN}") if f != "index.yaml" and f.endswith(".yaml")]
        #layers = sorted(l.split(".")[0] for l in layers)
    
        # # CiMLoop One Mapping
        #start_time_one_mapping = time.time()
    
    all_results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(run_layer)(DNN, layer, paramsLST,num_mapping)
        for DNN in dnn_names
        for layer in layers_allnets[DNN]
        #joblib.delayed(run_layer)(DNN, layer, avg_input, avg_weight, shape,paramsLST,1000)
        #for layer, avg_input, avg_weight, shape in zip(
        #    layers, input_averages, weight_averages, SHAPES
        #)
    )
    

    divided_list = divide_list_unequal(all_results, networks_sizes)
    #print(len(divided_list))
    for dnn_name,all_layers_lst in zip(dnn_names,divided_list):
        #print(dnn_name)
        #print(len(all_layers_lst))
        #n_layer_dnn=len(layers_allnets[dnn_name])
        total_energy=0
        total_area=0
        total_latency=0
        total_mac=0
        sum_tops_per_mm2=0
        sum_tops_per_w=0
        sum_computes_per_second=0
        sum_computes_per_second_per_square_meter=0
        max_percent_utilization=0
        max_tops_per_w=0
        max_tops_per_mm2=0
        countl=0
        tops=0
        for l in all_layers_lst:
            #print(attr(l))
            #print('areacheck',l.area*1e4)
            #print('l.percent_utilization',l.percent_utilization)
            tops+=l.tops
            total_area=l.area*1e4 #in cm2 
            total_energy+=l.energy #in J
            total_latency+=l.latency #in seconds
            total_mac+=l.computes
            sum_tops_per_mm2+=l.tops_per_mm2
            sum_tops_per_w+=l.tops_per_w
            sum_computes_per_second+=l.computes_per_second
            sum_computes_per_second_per_square_meter+=l.computes_per_second_per_square_meter
            max_percent_utilization=max(max_percent_utilization, l.percent_utilization)
            max_tops_per_w=max(max_tops_per_w,l.tops_per_w)
            max_tops_per_mm2=max(max_tops_per_mm2, l.tops_per_mm2)
            countl+=1
        #print('countl',countl)    
        #print('len(all_layers_lst)',len(all_layers_lst))
        avg_sum_tops_per_mm2=sum_tops_per_mm2/len(all_layers_lst)        
        avg_sum_tops_per_w=sum_tops_per_w/len(all_layers_lst)
        avg_computes_per_second=sum_computes_per_second/len(all_layers_lst)
        avg_computes_per_second_per_square_meter=sum_computes_per_second_per_square_meter/len(all_layers_lst)
        avg_tops=tops/len(all_layers_lst) 
        mydict[dnn_name]=[total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2,avg_tops]
        
    return mydict


def getMetrics(paramsLST,dnn_names,num_mapping,n_jobs):
    mydict={}
    for dnn_name in dnn_names:
        DNN = dnn_name
        layers = [f for f in os.listdir(f"../models/workloads/{DNN}") if f != "index.yaml" and f.endswith(".yaml")]
        layers = sorted(l.split(".")[0] for l in layers)
    
        # # CiMLoop One Mapping
        #start_time_one_mapping = time.time()
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(run_layer)(DNN, layer, paramsLST,num_mapping)
            for layer in layers
            #joblib.delayed(run_layer)(DNN, layer, avg_input, avg_weight, shape,paramsLST,1000)
            #for layer, avg_input, avg_weight, shape in zip(
            #    layers, input_averages, weight_averages, SHAPES
            #)
        )
        #end_time_one_mapping = time.time()
    
        #print('time per one:',end_time_one_mapping-start_time_one_mapping)
        total_energy=0
        total_area=0
        total_latency=0
        total_mac=0
        sum_tops_per_mm2=0
        sum_tops_per_w=0
        sum_computes_per_second=0
        sum_computes_per_second_per_square_meter=0
        max_percent_utilization=0
        max_tops_per_w=0
        max_tops_per_mm2=0
        tops=0
        for r in [results]:
            for l in r:
                total_area=l.area*1e4 #in cm2
                tops+=l.tops
                total_energy+=l.energy #in J
                total_latency+=l.latency #in seconds
                total_mac+=l.computes
                sum_tops_per_mm2+=l.tops_per_mm2
                sum_tops_per_w+=l.tops_per_w
                sum_computes_per_second+=l.computes_per_second
                sum_computes_per_second_per_square_meter+=l.computes_per_second_per_square_meter
                max_percent_utilization=max(max_percent_utilization, l.percent_utilization)
                max_tops_per_w=max(max_tops_per_w,l.tops_per_w)
                max_tops_per_mm2=max(max_tops_per_mm2, l.tops_per_mm2)
                
        avg_sum_tops_per_mm2=sum_tops_per_mm2/len(results)        
        avg_sum_tops_per_w=sum_tops_per_w/len(results)
        avg_computes_per_second=sum_computes_per_second/len(results)
        avg_computes_per_second_per_square_meter=sum_computes_per_second_per_square_meter/len(results)
        avg_tops=tops/len(results)
        mydict[dnn_name]=[total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2,avg_tops]
    
        #print('total_energy in J:',total_energy)
        #print('total_area in cm2:',total_area)
        #print('total_latency in s:',total_latency)
        #print('total_mac:',total_mac)
        #print('avg_sum_tops_per_mm2:',avg_sum_tops_per_mm2)
        #print('avg_sum_tops_per_w:',avg_sum_tops_per_w)
        #print('avg_computes_per_second',avg_computes_per_second)
        #print('avg_computes_per_second_per_square_meter',avg_computes_per_second_per_square_meter)
        #print('max_percent_utilization:',max_percent_utilization)
        #print('area_utilized in cm2:',max_percent_utilization*total_area)
        
        #print(results[0].mapping)
    return mydict


def checkIfallDictExist(dnn_names,file_store_paths, datakey):
    for dnn_name in dnn_names:
        dict_load=load_dict_from_json(file_store_paths[dnn_name])
        if search_instance_in_dict(dict_load, datakey)==False:
            a=False
            break
        if dnn_name == dnn_names[-1]:
            a=True
    if a==True:
        return True
    else:
        return False
    

def run_several_iterationsJoint(dnn_names, num_generations1, population_size1, num_iterations_to_run, seed_alg1, areaConstrcm2,dir_noDNNname,obj_type,num_mapping,n_jobs,samplesToEval,samplesToEvalTest,prob_cross1, prob_mute1, eta_cross1, eta_mute1, num_generations2, population_size2,prob_cross2, prob_mute2, eta_cross2, eta_mute2, num_generations3, population_size3,prob_cross3, prob_mute3, eta_cross3, eta_mute3, num_generations4, population_size4,prob_cross4, prob_mute4, eta_cross4, eta_mute4):
    if seed_alg1==None:
        seed_alg=None
    else:
        seed_alg=seed_alg1
    for testnumber in range(num_iterations_to_run):
        print('Iteration', testnumber)
        #testnumber=0
        directorys={}
        resultsfiles={}
        resultsfiles2={}
        file_store_paths={}
        file_store_pathsOnlyTrue={}
        
        
        #fileStat='res/NeuroSimLikeJointAreaConst/'+'AllStat_'+'test'+str(testnumber)+'.json'
        #fileStatFilt='res/NeuroSimLikeJointAreaConst/'+'AllStatFilt_'+'test'+str(testnumber)+'.json'
        fileMeanStat=dir_noDNNname+'MeanStat_'+'test'+str(testnumber)+'.json'    
        for dnn_name in dnn_names:
            #directory = 'res/'+'NeuroSimLikeJointAreaConst_Latency/'+dnn_name
            directory=dir_noDNNname+dnn_name
            directorys[dnn_name]=directory
            resultsfile='test'+str(testnumber)+'.json'
            resultsfile2='test'+str(testnumber)+'Constr'+'.json'
            resultsfiles[dnn_name]=resultsfile
            resultsfiles2[dnn_name]=resultsfile2
            file_store_path=directory+'/'+resultsfile
            file_store_path2=directory+'/'+resultsfile2
            file_store_paths[dnn_name]=file_store_path
            file_store_pathsOnlyTrue[dnn_name]=file_store_path2
            # Create the directory if it does not exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            empty_dict = {}
            # Store the empty dictionary to a JSON file
            with open(file_store_path, 'w') as f:
                json.dump(empty_dict, f, indent=4)
            
            with open(file_store_path2, 'w') as f:
                json.dump(empty_dict, f, indent=4)
            
        
        # Create the file with an empty list if it doesn't exist
        #with open(fileStat, 'w') as file:
        #    json.dump([], file)
        # Create the file with an empty list if it doesn't exist
        with open(fileMeanStat, 'w') as file:
            json.dump([], file)

        #with open(fileStatFilt, 'w') as file:
        #    json.dump([], file)
        
        coeff_lat=1.
        coeff_en=1.
        coeff_ar=1.
        
        def objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type):
            if obj_type=='l':
                obj=max(latencys)
            elif obj_type=='e':
                obj=max(energys)
            elif obj_type=='a':
                obj=area
            elif obj_type=='el':
                obj=max(energys)*max(latencys)
            elif obj_type=='ela':
                obj=max(energys)*max(latencys)*area
            elif obj_type=='ela_mean':
                obj=(sum(energys)/len(energys))*(sum(latencys)/len(latencys))*area
            #return math.log(pow(latency, coeff_lat)*pow(energy, coeff_en)*pow(area, coeff_ar))
            #obj=max(energys)*max(latencys)*area
            #obj=max(latencys)
            return obj
        
        layers_all={}
        for dnn_name in dnn_names:
            layers = [f for f in os.listdir(f"../models/workloads/{dnn_name}") if f != "index.yaml" and f.endswith(".yaml")]
            layers = sorted(l.split(".")[0] for l in layers)
            layers_all[dnn_name]=layers
        
        
        voltages = [x / 100 for x in range(50, 90, 5)] #8
        #bits_per_cell=[1]
        base_latency = [1e-9,2e-9,4e-9,6e-9,8e-9,1e-8]
        crossbar_sizeX=[16, 32, 64, 128, 256, 512] #for the main crossbar
        #crossbar_sizeY=[32, 64, 128, 256, 512]
        crossbar_sizeY=[16, 32, 64, 128, 256, 512]
        shared_router_groupSize=[1,2,4,6,8,16,32]
        #router_width=[32,64,128] #???? --> doesn't change anything
        #tiles_in_chip=[4,8,16,32,64,128]
        tiles_in_chip=[2,4,8,16,32,64]
        #macros_in_tile=[4,8,16,32,64,128,256]
        macros_in_tile=[2,4,8,16,32,64,128]
        #macros_in_tile=[1]
        #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [2,4,8,16,32,64,128]]
        glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [1,2,4,8,16,32,64,128]]
        


        
        inp_bits=8
        
        def getCBsize(lst,dnn_name,layer_name):
            designspecs=getSpecs(dnn_name,layer_name)
            ins=designspecs.problem.instance
            C,R,S,M=ins["C"],ins["R"],ins["S"],ins["M"]
            rows=C*R*S
            #cols=M*math.ceil(inp_bits/bits_per_cell[lst[1]])
            cols=M*math.ceil(inp_bits)
            total_cb=rows*cols
            return total_cb
        
        def checkSizeParallel(lst,dnn_name,layers,inp_bits,n_jobs):
            total_cb=0

            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(getCBsize)(lst,dnn_name,layer_name)
                for layer_name in layers
            )
            total_cb=sum(results)
            actual_cbs=crossbar_sizeX[lst[2]]*crossbar_sizeY[lst[3]]*shared_router_groupSize[lst[4]]*tiles_in_chip[lst[5]]*macros_in_tile[lst[6]]
            if actual_cbs>=total_cb:
                return True
            else:
                return False
        
        def checkSize(lst,dnn_name,layers,inp_bits):
            total_cb=0
            for layer_name in layers:
                designspecs=getSpecs(dnn_name,layer_name)
                ins=designspecs.problem.instance
                C,R,S,M=ins["C"],ins["R"],ins["S"],ins["M"]
                rows=C*R*S
                #cols=M*math.ceil(inp_bits/bits_per_cell[lst[1]])
                cols=M*math.ceil(inp_bits)
                total_cb+=rows*cols
            actual_cbs=crossbar_sizeX[lst[2]]*crossbar_sizeY[lst[3]]*shared_router_groupSize[lst[4]]*tiles_in_chip[lst[5]]*macros_in_tile[lst[6]]
            #print('actual_cbs',actual_cbs)
            #print('total_cb',total_cb)
            if actual_cbs>=total_cb:
                return True
            else:
                return False
        
        def total_required(lst,dnn_name,layers,inp_bits):
            total_cb=0
            for layer_name in layers:
                designspecs=getSpecs(dnn_name,layer_name)
                ins=designspecs.problem.instance
                C,R,S,M=ins["C"],ins["R"],ins["S"],ins["M"]
                rows=C*R*S
                #cols=M*math.ceil(inp_bits/bits_per_cell[lst[1]])
                cols=M*math.ceil(inp_bits)
                total_cb+=rows*cols
            return total_cb  
            
        def total_requiredParallel(lst,dnn_name,layers,inp_bits,n_jobs):
            total_cb=0
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(getCBsize)(lst,dnn_name,layer_name)
                for layer_name in layers
            )
            total_cb=sum(results)
            return total_cb  
            
        def actual_present(lst,dnn_name,layers,inp_bits):
            actual_cbs=crossbar_sizeX[lst[2]]*crossbar_sizeY[lst[3]]*shared_router_groupSize[lst[4]]*tiles_in_chip[lst[5]]*macros_in_tile[lst[6]]
            return actual_cbs 
        #    valid=[]
        #for lists in listp:
        #    actual_cbs=lists[3]*lists[4]*lists[7]*lists[6]*lists[5]
        #    if actual_cbs>=total_cb:
        #        valid.append(lists)
        #print(len(valid))
        #print(valid[0])
        #actual_cbs=crossbar_sizeX*crossbar_sizeY*macros_in_tile*tiles_in_chip*shared_router_groupSize*macros_in_chip
        #print(actual_cbs)
        
        def eliminateWrongSizesParallel(x,dnn_name,layers,n_jobs):
            final_output=[]
            #print(np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]))
            for i in range(x.shape[0]):
                check=checkSizeParallel(x[i],dnn_name,layers,inp_bits,n_jobs)
                if check==True:
                    final_output.append(x[i])
            
            n, (xl, xu) = problem.n_var, problem.bounds()
            
            #print('***Sampling')
            #print('output:')
            #final_output=[]
            #print(np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]))
            #print('Replaced number:', x.shape[0]-len(final_output))
            while len(final_output)<x.shape[0]:
                fullfilled=len(final_output)
                output= np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=x.shape[0]-fullfilled) for k in range(n)])
                for i in range(output.shape[0]):
                    check=checkSizeParallel(output[i],dnn_name,layers,inp_bits,n_jobs)
                    if check==True:
                        final_output.append(output[i])
            return np.row_stack([final_output[kk] for kk in range(len(final_output))])
        
        def eliminateWrongSizes(x,dnn_name,layers):
            final_output=[]
            #print(np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]))
            for i in range(x.shape[0]):
                check=checkSize(x[i],dnn_name,layers,inp_bits)
                if check==True:
                    final_output.append(x[i])
            
            n, (xl, xu) = problem.n_var, problem.bounds()
            
            #print('***Sampling')
            #print('output:')
            #final_output=[]
            #print(np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]))
            #print('Replaced number:', x.shape[0]-len(final_output))
            while len(final_output)<x.shape[0]:
                fullfilled=len(final_output)
                output= np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=x.shape[0]-fullfilled) for k in range(n)])
                for i in range(output.shape[0]):
                    check=checkSize(output[i],dnn_name,layers,inp_bits)
                    if check==True:
                        final_output.append(output[i])
            return np.row_stack([final_output[kk] for kk in range(len(final_output))])
        
        class MySamplingSeveralX(FloatRandomSampling):
            def _do(self, problem, n_samples, **kwargs):
                n, (xl, xu) = problem.n_var, problem.bounds()
                #print('***Sampling')
                #print('output:')
                final_output=[]
                #print(np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]))
                while len(final_output)<n_samples:
                    fullfilled=len(final_output)
                    output= np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples-fullfilled) for k in range(n)])
                    for i in range(output.shape[0]):
                        final_output.append(output[i])
                #print(np.row_stack([final_output[kk] for kk in range(len(final_output))]))
                #print(final_output)
                #print(len(final_output))
                return np.row_stack([final_output[kk] for kk in range(len(final_output))])
        

        def hamming_distance_indices(design_a, design_b):
            """
            Calculate the Hamming distance based on the indices where the values differ.
        
            Parameters:
            design_a (list): First design represented as a list of discrete values.
            design_b (list): Second design represented as a list of discrete values.
        
            Returns:
            int: Hamming distance (number of differing indices).
            list: List of indices where the values differ.
            """
            if len(design_a) != len(design_b):
                raise ValueError("Designs must be of the same length")
        
            # Find indices where values differ
            differing_indices = [i for i in range(len(design_a)) if design_a[i] != design_b[i]]
        
            # Hamming distance is the count of differing indices
            hamming_distance = len(differing_indices)
        
            return hamming_distance, differing_indices
        
        def select_max_hamming_designs2(designs, k, n_jobs):
            """
            Select k designs with near-maximum overall Hamming distance using a greedy approach.
        
            Parameters:
                designs (list): List of designs.
                k (int): Number of designs to select.
        
            Returns:
                tuple: (np.array(selected designs), total Hamming distance)
            """
            selected = [designs[0]]  # Start with the first design or choose randomly
            
            while len(selected) < k:
                best_candidate = None
                best_min_distance = -1
        
                # Evaluate each design not already selected
                for candidate in designs:
                    # Use np.array_equal to check if candidate is already selected
                    if any(np.array_equal(candidate, s) for s in selected):
                        continue
        
                    # Compute the minimum Hamming distance from candidate to all selected designs
                    min_distance = min(hamming_distance_indices(candidate, s)[0] for s in selected)
        
                    # Select the candidate with the largest minimum distance
                    if min_distance > best_min_distance:
                        best_min_distance = min_distance
                        best_candidate = candidate
        
                if best_candidate is None:
                    break  # No further candidate found; break out
        
                selected.append(best_candidate)
        
            # Compute total pairwise Hamming distance among selected designs
            total_distance = sum(
                hamming_distance_indices(selected[i], selected[j])[0]
                for i in range(len(selected))
                for j in range(i + 1, len(selected))
            )
        
            return np.array(selected), total_distance

        
       

        
        def select_max_hamming_designsCrashes(designs, k, n_jobs):
            """
            Select k designs with the maximum overall Hamming distance.
        
            Parameters:
            designs (list of lists): List containing multiple designs.
            k (int): Number of designs to select.
            n_jobs (int): The number of parallel jobs to run. -1 uses all available processors.
        
            Returns:
            list: Selected designs with maximum Hamming distance.
            """
            def compute_total_distance(combo):
                total_distance = 0
                for i in range(len(combo)):
                    for j in range(i + 1, len(combo)):
                        distance, _ = hamming_distance_indices(combo[i], combo[j])
                        total_distance += distance
                return total_distance, combo
        
            # Generate all combinations of k designs
            combos = list(combinations(designs, k))
            
            # Parallelize the computation of total distances
            results = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(compute_total_distance)(combo) for combo in combos
            )
        
            # Find the combination with the maximum total distance
            max_total_distance, selected_designs = max(results, key=lambda x: x[0])
        
            return np.array(selected_designs), max_total_distance
        
        
        def select_max_hamming_designs(designs, k):
            """
            Select k designs with the maximum overall Hamming distance.
        
            Parameters:
            designs (list of lists): List containing multiple designs.
            k (int): Number of designs to select.
        
            Returns:
            list: Selected designs with maximum Hamming distance.
            """
            max_total_distance = -1
            selected_designs = []
        
            # Generate all combinations of k designs
            for combo in combinations(designs, k):
                total_distance = 0
                for i in range(len(combo)):
                    for j in range(i + 1, len(combo)):
                        distance, _ = hamming_distance_indices(combo[i], combo[j])
                        total_distance += distance
        
                if total_distance > max_total_distance:
                    max_total_distance = total_distance
                    selected_designs = combo
        
            return np.array(selected_designs), max_total_distance
        
        
        def sort_designs_by_score(design_params, scores):
            """
            Sorts the design parameters based on their corresponding scores in ascending order.
        
            Parameters:
            design_params (np.ndarray): An array of design parameters (shape: [n_samples, n_features]).
            scores (np.ndarray): A 1D array of corresponding scores (shape: [n_samples]).
        
            Returns:
            np.ndarray: Sorted design parameters based on minimum score.
            np.ndarray: Sorted scores in ascending order.
            """
            # Get the indices that would sort the scores in ascending order
            sorted_indices = np.argsort(scores)
        
            # Sort design parameters and scores using the sorted indices
            sorted_designs = design_params[sorted_indices]
            sorted_scores = scores[sorted_indices]
        
            return sorted_designs, sorted_scores

        
        def convert_config_to_coeffs(config_str):
            # Define the design parameter lists
            #voltages = [x / 100 for x in range(50, 90, 5)]         # [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
            #bits_per_cell = [1, 2, 3, 4, 5, 6, 7, 8]                # Coefficients: 0 through 7
            #base_latency = [1e-9, 2e-9, 4e-9, 6e-9, 8e-9, 1e-8]     # Coefficients: 0 through 5
            #crossbar_sizeX = [32, 64, 128, 256, 512]                # Coefficients: 0 through 4
            #crossbar_sizeY = [32, 64, 128, 256, 512]                # Coefficients: 0 through 4
            #shared_router_groupSize = [1, 2, 4, 8, 16, 32]         # Coefficients: 0 through 5
            #tiles_in_chip = [4, 8, 16, 32, 64]                     # Coefficients: 0 through 4
            #macros_in_tile = [4, 8, 16, 32, 64, 128]               # Coefficients: 0 through 5
            #glb_buffer_depth = [int(1024 * 1024 * x * 8 / 2048) for x in [0.5, 1, 2, 4, 8, 16, 32]]
            # This computes to: [2048, 4096, 8192, 16384, 32768, 65536, 131072]
        
            # Split the configuration string by underscores.
            # The configuration string format is:
            # "A_voltage_bits_latency_crossbarX_crossbarY_router_tiles_macros_buffer"
            parts = config_str.split('_')
        
            # Convert parts to appropriate types (skip the first design label 'A')
            voltage = float(parts[1])
            #bits = int(parts[2])
            latency = float(parts[3])
            x_size = int(parts[4])
            y_size = int(parts[5])
            router_size = int(parts[6])
            tiles = int(parts[7])
            macros = int(parts[8])
            buffer = int(parts[9])
        
            # Get the coefficient (index) for each value in its respective list
            voltage_coeff = voltages.index(voltage)
            #bits_coeff = bits_per_cell.index(bits)
            latency_coeff = base_latency.index(latency)
            x_coeff = crossbar_sizeX.index(x_size)
            y_coeff = crossbar_sizeY.index(y_size)
            router_coeff = shared_router_groupSize.index(router_size)
            tiles_coeff = tiles_in_chip.index(tiles)
            macros_coeff = macros_in_tile.index(macros)
            buffer_coeff = glb_buffer_depth.index(buffer)
        
            # Return the coefficients as a list in the defined order:
            # [voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth]
            return [voltage_coeff, latency_coeff, x_coeff, y_coeff, router_coeff, tiles_coeff, macros_coeff, buffer_coeff]
        
        # Example usage:
        #config_str = "A_0.7_8_6e-09_512_64_2_64_64_4096"
        #coefficients = convert_config_to_coeffs(config_str)
        #print(coefficients)

        def getBestDesignsFromDict(file_store_pathsTHIS):
            dataNEW={}
            #file_store_paths[dnn_name]=file_store_path
            #file_store_pathsOnlyTrue[dnn_name]=file_store_path2
            
            for dnn in dnn_names:
                pathNEW=file_store_pathsTHIS[dnn_name] #file_store_pathsOnlyTrue[dnn_name]
                dataNEW[dnn]=load_dict_from_json(pathNEW)
                if dnn==dnn_names[-1]:
                    keys_listNEW = list(dataNEW[dnn].keys())
            scoresNEW={}
            for keys in keys_listNEW:
                latencysNEW=[dataNEW[dnn][keys]['l'] for dnn in dnn_names]
                energysNEW=[dataNEW[dnn][keys]['e'] for dnn in dnn_names]
                areasNEW=[dataNEW[dnn][keys]['a'] for dnn in dnn_names]
                areaNEW=max(areasNEW)
                scoresNEW[keys]=objectiveMAX(latencysNEW, energysNEW, areaNEW, coeff_lat, coeff_en, coeff_ar,obj_type)
                
            top_NEW = sorted(scoresNEW.items(), key=lambda x: x[1], reverse=False)
            return top_NEW
        
        class MySamplingSeveralParallelPhaseTwo(FloatRandomSampling):
            def _do(self, problem, n_samples, **kwargs):
                start_time_sampl = time.time()
                n, (xl, xu) = problem.n_var, problem.bounds()
                #print('***Sampling')
                #print('output:')
                final_output=[]
                #print(np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]))
                
                #top_constraint=getBestDesignsFromDict(file_store_pathsOnlyTrue)
                #numDesingsConstr=len(top_constraint)

                top_constraint = getBestDesignsFromDict(file_store_pathsOnlyTrue)
                num_designs_constr = len(top_constraint)
            
                sampled_designs = []
            
                if num_designs_constr >= n_samples:
                    # If we have enough constrained designs, take the first n_samples
                    sampled_designs = top_constraint[:n_samples]
                else:
                    # Use all constrained designs available
                    sampled_designs = top_constraint[:]
                    remaining = n_samples - len(sampled_designs)

                    unique_constraint_keys = {design_key for design_key, _ in top_constraint}
                    top_unconstraint = getBestDesignsFromDict(file_store_paths)
                    top_unconstraint_unique = [
                        (design_key, score)
                        for design_key, score in top_unconstraint
                        if design_key not in unique_constraint_keys
                    ]
                    
                    # Get designs from the "unconstrained" dictionary
                    #top_unconstraint = getBestDesignsFromDict(file_store_paths)
                    #num_designs_unconstr = len(top_unconstraint)
                    
                    if len(top_unconstraint_unique) >= remaining:
                        # If there are enough unconstrained designs to fill the gap, take as many as needed
                        sampled_designs += top_unconstraint[:remaining]
                    else:
                        # Use all unconstrained designs and update the number still needed
                        sampled_designs += top_unconstraint_unique
                        remaining = n_samples - len(sampled_designs)
                        
                        # If still not enough, generate random designs from the design space
                        # Define the design parameter lists
                        #voltages = [x / 100 for x in range(50, 90, 5)]         # [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
                        #bits_per_cell = [1, 2, 3, 4, 5, 6, 7, 8]                # Coefficients: 0-7
                        #base_latency = [1e-9, 2e-9, 4e-9, 6e-9, 8e-9, 1e-8]     # Coefficients: 0-5
                        #crossbar_sizeX = [32, 64, 128, 256, 512]                # Coefficients: 0-4
                        #crossbar_sizeY = [32, 64, 128, 256, 512]                # Coefficients: 0-4
                        #shared_router_groupSize = [1, 2, 4, 8, 16, 32]         # Coefficients: 0-5
                        #tiles_in_chip = [4, 8, 16, 32, 64]                     # Coefficients: 0-4
                        #macros_in_tile = [4, 8, 16, 32, 64, 128]               # Coefficients: 0-5
                        #glb_buffer_depth = [int(1024 * 1024 * x * 8 / 2048) for x in [0.5, 1, 2, 4, 8, 16, 32]]
                        # This computes to: [2048, 4096, 8192, 16384, 32768, 65536, 131072]
                        
                        existing_keys = {design_key for design_key, _ in sampled_designs}
                        
                        # Generate random designs until we have the required number of samples.
                        while len(sampled_designs) < n_samples:
                            voltage = random.choice(voltages)
                            #bits = random.choice(bits_per_cell)
                            bits = 1
                            latency = random.choice(base_latency)
                            x_size = random.choice(crossbar_sizeX)
                            y_size = random.choice(crossbar_sizeY)
                            router_size = random.choice(shared_router_groupSize)
                            tiles = random.choice(tiles_in_chip)
                            macros = random.choice(macros_in_tile)
                            buffer = random.choice(glb_buffer_depth)
                            
                            # Construct the design key string in the expected format.
                            design_key = f"A_{voltage}_{bits}_{latency}_{x_size}_{y_size}_{router_size}_{tiles}_{macros}_{buffer}"
                            
                            # Only add the candidate if it hasn't been seen before.
                            if design_key in existing_keys:
                                continue  # Skip duplicate keys.
                            
                            sampled_designs.append((design_key, 0))  # Append with a dummy score.
                            existing_keys.add(design_key)
            
                # Convert all design keys to coefficient vectors using the helper function.
                samples = [convert_config_to_coeffs(design_key) for design_key, _ in sampled_designs]
                
                end_time_sampl= time.time()
                
                print('Sampling finished in time in s:', end_time_sampl-start_time_sampl)

                return np.array(samples)
                
                #toreturn=np.row_stack([final_output[kk] for kk in range(len(final_output))])
               # print(toreturn)
                #print(type(toreturn))
                #selected_designs, max_total_distance = select_max_hamming_designs2(toreturn, k=n_samples,n_jobs=n_jobs)
                #print('selected_designs',selected_designs)
                #print('max_total_distance',max_total_distance)
                #print(type(selected_designs))
                
                #return selected_designs
                
        
        
        class MySamplingSeveralParallelHammingDistWithEval(FloatRandomSampling):
            def _do(self, problem, n_samples, **kwargs):
                start_time_sampl = time.time()
                n, (xl, xu) = problem.n_var, problem.bounds()
                #print('***Sampling')
                #print('output:')
                #final_output=[]
                final_output = np.column_stack([
                    np.random.randint(xl[k], xu[k] + 1, size=samplesToEval) for k in range(n)]).tolist()

                #print(np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]))
                #while len(final_output)<samplesToEval:
                #    fullfilled=len(final_output)
                #    print('Sampled true designs remaining',samplesToEval-fullfilled)
                #    output= np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=samplesToEval-fullfilled) for k in range(n)])
                    
                 #   for i in range(output.shape[0]):
                 #       for dnn_name in dnn_names:
                            #print(output[i])
                  #          check=checkSizeParallel(output[i],dnn_name,layers_all[dnn_name],inp_bits,n_jobs)
                  #          if check==False:
                  #              break
                  #      if check==True: #if we reach the last check and it's true
                  #          final_output.append(output[i])
                #print(np.row_stack([final_output[kk] for kk in range(len(final_output))]))
                #print(final_output)
                #print(len(final_output))
                end_time_sampl= time.time()
                print('Sampling finished in time in s:', end_time_sampl-start_time_sampl)


                toreturn=np.row_stack([final_output[kk] for kk in range(len(final_output))])
                #print('Designs before Hamming',toreturn)
                #print(type(toreturn))
                selected_designs, max_total_distance = select_max_hamming_designs2(toreturn, k=samplesToEvalTest, n_jobs=n_jobs)
                #print('selected_designs after Hamming',selected_designs)
                #print('max_total_distance',max_total_distance)
                #print(type(selected_designs))

                print('Designs with Hamming distance selected in time in s:', end_time_sampl-start_time_sampl)

                new_xS=np.zeros(selected_designs.shape)
                new_fS=np.zeros(selected_designs.shape[0])
                #print(range(x.shape[0]))
                for i in range(selected_designs.shape[0]):
                    #print(x[i])
                    #print(C[x[i][0]])
                    #print(Y[x[i][1]])
                    new_xS[i][0]=voltages[selected_designs[i][0]]
                    #new_xS[i][1]=bits_per_cell[selected_designs[i][1]]
                    new_xS[i][1]=base_latency[selected_designs[i][1]]
                    new_xS[i][2]=crossbar_sizeX[selected_designs[i][2]]
                    new_xS[i][3]=crossbar_sizeY[selected_designs[i][3]]
                    new_xS[i][4]=shared_router_groupSize[selected_designs[i][4]]
                    new_xS[i][5]=tiles_in_chip[selected_designs[i][5]]
                    new_xS[i][6]=macros_in_tile[selected_designs[i][6]]
                    new_xS[i][7]=glb_buffer_depth[selected_designs[i][7]]

                    


                for i in range(new_xS.shape[0]):
                    #print('i evaluation in Sampling',i)
                    #print(' ')
                    #print(new_x[i])
                    #total_energy, total_area, total_latency=getMetrics(new_x[i])
                    #score=objective(total_latency, total_energy, total_area, coeff_lat, coeff_en, coeff_ar)
                    
                    #latency, energy, area=get_metrics(dic, new_x[i][0], new_x[i][1], new_x[i][2], new_x[i][3], new_x[i][4]) 
                    #score=objective(latency, energy, area, coeff_lat, coeff_en, coeff_ar)
                    
                    #score_check(dic_scores, new_x[i][0], new_x[i][1], new_x[i][2], new_x[i][3], new_x[i][4], score)
                    try:
                        
                        #vl,bps,bl,csX,csY,srg,tic,mit,gbp=new_xS[i]
                        vl,bl,csX,csY,srg,tic,mit,gbp=new_xS[i]
                        bps=1
                        vl,bl =[float(x) for x in [vl,bl]]
                        write_to_dic=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        bps,csX,csY,srg,tic,mit,gbp=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        write_to_dic=vl,bps,bl,csX,csY,srg,tic,mit,gbp
                        datakey='A_'+'_'.join(map(str, write_to_dic))
                        #print('datakey',datakey)
                        energies={}
                        areas={}
                        latencies={}
                        if checkIfallDictExist(dnn_names,file_store_paths, datakey)==True:
                            for dnn_name in dnn_names:
                                dict_load=load_dict_from_json(file_store_paths[dnn_name])
                                #if search_instance_in_dict(dict_load, datakey)==True:
                                dataForScore=dict_load[datakey] # dictionary with data
                                total_latency=dataForScore['l']
                                total_energy=dataForScore['e']
                                total_area=dataForScore['a']
                                max_percent_utilization=dataForScore['u']
                                avg_sum_tops_per_mm2=dataForScore['a_tpm']
                                avg_sum_tops_per_w=dataForScore['a_tpw']
                                max_tops_per_w=dataForScore['m_tpw']
                                max_tops_per_mm2=dataForScore['m_tpm']
                                ##???: correct?
                                
                                energies[dnn_name]=total_energy
                                areas[dnn_name]=total_area
                                latencies[dnn_name]=total_latency
                        ############################
                        else:
                        
                            
                            dictRes=getMetricsParallel(new_xS[i],dnn_names,num_mapping,n_jobs)
                            #energies={}
                            #areas={}
                            #latencies={}
                            dictts={}
                            #constrGood=True
                            for dnn_name in dnn_names:
                                #total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2=dictRes[dnn_name]
                                total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2,tops=dictRes[dnn_name]
                                
                                energies[dnn_name]=total_energy
                                areas[dnn_name]=total_area
                                latencies[dnn_name]=total_latency
                                #dictts[dnn_name]={'e': total_energy, 'a': total_area, 'l':total_latency,'a_tpm':avg_sum_tops_per_mm2, 'a_tpw': avg_sum_tops_per_w,'u': max_percent_utilization,'au': max_percent_utilization*total_area, 'm_tpm':max_tops_per_mm2, 'm_tpw':max_tops_per_w }
                                
                                dictts[dnn_name]={'e': total_energy, 'a': total_area, 'l':total_latency,'a_tpm':avg_sum_tops_per_mm2, 'a_tpw': avg_sum_tops_per_w,'u': max_percent_utilization,'au': max_percent_utilization*total_area, 'm_tpm':max_tops_per_mm2, 'm_tpw':max_tops_per_w,'tp':tops }
                                
                                add_new_instance_if_not_exists(file_store_paths[dnn_name], datakey, dictts[dnn_name])
                    
                                #total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2=getMetrics(new_x[i])
                                #print('********Network:',dnn_name)
                                #print('total_energy in J:',total_energy)
                                #print('total_area in cm2:',total_area)
                                #print('total_latency in s:',total_latency)
                                #print('avg_sum_tops_per_mm2:',avg_sum_tops_per_mm2)
                                #print('avg_sum_tops_per_w:',avg_sum_tops_per_w)
                                #print('max_percent_utilization:',max_percent_utilization)
                                #print('area_utilized in cm2:',max_percent_utilization*total_area)
                                #print('max_tops_per_w:',max_tops_per_w)
                                #print('max_tops_per_mm2:',max_tops_per_mm2)
                            areaslistconst=[]
                            for dnn_name in dnn_names:
                                areaslistconst.append(areas[dnn_name])
                            all_good = all(x <= areaConstrcm2 for x in areaslistconst)
                            if all_good==True:
                                for dnn_name in dnn_names:
                                    add_new_instance_if_not_exists(file_store_pathsOnlyTrue[dnn_name], datakey, dictts[dnn_name])
                                
                        latlist=list(latencies.values())
                        englist=list(energies.values())
                        arlist=list(areas.values())
                        if all(x == arlist[0] for x in arlist)==False:
                            print('not all areas are the same --> Check!')
                        areaforscore=np.mean(arlist)
                        score=objectiveMAX(latlist, englist, areaforscore, coeff_lat, coeff_en, coeff_ar,obj_type)    
                    #score=0.5

                    #### THE REST FINISHED HERE!!!!! continue:
                    
                    except RuntimeError:
                        #vl,bps,bl,csX,csY,srg,tic,mit,gbp=new_xS[i]
                        vl,bl,csX,csY,srg,tic,mit,gbp=new_xS[i]
                        bps=1
                        vl,bl =[float(x) for x in [vl,bl]]
                        write_to_dic=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        bps,csX,csY,srg,tic,mit,gbp=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        write_to_dic=vl,bps,bl,csX,csY,srg,tic,mit,gbp
                        datakey='A_'+'_'.join(map(str, write_to_dic))
                        #print('datakey',datakey)
                        score=1e25 #setting for very large score (just in case)
                        areaforscore=1e25
                        #print('Runtime error --> mutation produced too small system')
                        continue
                    
                    #areaConstrcm2
                    new_fS[i]=score

                end_time_sampl= time.time()
                print('Sampling - evaluation finished in time in s:', end_time_sampl-start_time_sampl)

                
                #print('Sorting desings:')
                #print('selected_designs',selected_designs)
                #print('new_fS',new_fS)
                sorted_designs, sorted_scores = sort_designs_by_score(selected_designs, new_fS)
                #print("Sorted Designs after Eval:\n", sorted_designs)
                #print("Sorted Scores after Eval:\n", sorted_scores)
                
                # Get the best 3 designs
                best_designs = sorted_designs[:n_samples]
                #print("best_designs - to return:\n", sorted_designs)
                return best_designs
        
        class MySamplingSeveralParallel(FloatRandomSampling):
            def _do(self, problem, n_samples, **kwargs):
                start_time_sampl = time.time()
                n, (xl, xu) = problem.n_var, problem.bounds()
                #print('***Sampling')
                #print('output:')
                final_output=[]
                #print(np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]))
                while len(final_output)<n_samples:
                    
                    fullfilled=len(final_output)
                    print('remaining to sample:',n_samples-fullfilled)
                    output= np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples-fullfilled) for k in range(n)])
                    for i in range(output.shape[0]):
                        for dnn_name in dnn_names:
                            check=checkSizeParallel(output[i],dnn_name,layers_all[dnn_name],inp_bits,n_jobs)
                            if check==False:
                                break
                        if check==True: #if we reach the last check and it's true
                            final_output.append(output[i])
                #print(np.row_stack([final_output[kk] for kk in range(len(final_output))]))
                #print(final_output)
                #print(len(final_output))
                end_time_sampl= time.time()
                print('Sampling finished in time in s:', end_time_sampl-start_time_sampl)
                return np.row_stack([final_output[kk] for kk in range(len(final_output))])
        
        class MySamplingSeveral(FloatRandomSampling):
            def _do(self, problem, n_samples, **kwargs):
                start_time_sampl = time.time()
                n, (xl, xu) = problem.n_var, problem.bounds()
                #print('***Sampling')
                #print('output:')
                final_output=[]
                #print(np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples) for k in range(n)]))
                while len(final_output)<n_samples:
                    fullfilled=len(final_output)
                    output= np.column_stack([np.random.randint(xl[k], xu[k] + 1, size=n_samples-fullfilled) for k in range(n)])
                    for i in range(output.shape[0]):
                        for dnn_name in dnn_names:
                            check=checkSize(output[i],dnn_name,layers_all[dnn_name],inp_bits)
                            if check==False:
                                break
                        if check==True: #if we reach the last check and it's true
                            final_output.append(output[i])
                #print(np.row_stack([final_output[kk] for kk in range(len(final_output))]))
                #print(final_output)
                #print(len(final_output))
                end_time_sampl= time.time()
                print('Sampling finished in time in s:', end_time_sampl-start_time_sampl)
                return np.row_stack([final_output[kk] for kk in range(len(final_output))])

        #class MyProblem(ElementwiseProblem):
        class MyProblemParallel(Problem):
        
            def __init__(self):
                #super().__init__(n_var=2, n_obj=1, n_ieq_constr=0, xl=[0, 0], xu=[3, 2], vtype=int)
                #7,5,4,3,5,4,5,6, # for hardware
                
                super().__init__(n_var=8, n_obj=1, n_ieq_constr=10, xl=[0,0,0,0,0,0,0,0], xu=[7,5,5,5,6,5,6,7], vtype=int,elementwise_evaluation=True)
        
            def _evaluate(self, x, out, *args, **kwargs):
                #print('Before')
                #print(x)
                #newxxx=eliminateWrongSizes(x)
                #print('After')
               # print(newxxx)
                #print('**********x')
                #print(x)
                new_x=np.zeros(x.shape)
                new_f=np.zeros(x.shape[0])
                constrs=10
                new_g=np.zeros((x.shape[0],constrs))
                #print(range(x.shape[0]))
                for i in range(x.shape[0]):
                    #print(x[i])
                    #print(C[x[i][0]])
                    #print(Y[x[i][1]])
                    new_x[i][0]=voltages[x[i][0]]
                    #new_x[i][1]=bits_per_cell[x[i][1]]
                    new_x[i][1]=base_latency[x[i][1]]
                    new_x[i][2]=crossbar_sizeX[x[i][2]]
                    new_x[i][3]=crossbar_sizeY[x[i][3]]
                    new_x[i][4]=shared_router_groupSize[x[i][4]]
                    new_x[i][5]=tiles_in_chip[x[i][5]]
                    new_x[i][6]=macros_in_tile[x[i][6]]
                    new_x[i][7]=glb_buffer_depth[x[i][7]]
                #new_x=np.asarray(new_x, dtype = 'int')
                
                #print('new_x.shape',new_x.shape)
                #print('new_x.shape[0]',new_x.shape[0])
                #print('new_x[0]',new_x[0])
                #print('new_x[19]',new_x[19])
                for i in range(new_x.shape[0]):
                    #print(' ')
                    #print(new_x[i])
                    #total_energy, total_area, total_latency=getMetrics(new_x[i])
                    #score=objective(total_latency, total_energy, total_area, coeff_lat, coeff_en, coeff_ar)
                    
                    #latency, energy, area=get_metrics(dic, new_x[i][0], new_x[i][1], new_x[i][2], new_x[i][3], new_x[i][4]) 
                    #score=objective(latency, energy, area, coeff_lat, coeff_en, coeff_ar)
                    
                    #score_check(dic_scores, new_x[i][0], new_x[i][1], new_x[i][2], new_x[i][3], new_x[i][4], score)
                    try:
                        
                        vl,bl,csX,csY,srg,tic,mit,gbp=new_x[i]
                        bps=1
                        vl,bl =[float(x) for x in [vl,bl]]
                        write_to_dic=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        bps,csX,csY,srg,tic,mit,gbp=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        write_to_dic=vl,bps,bl,csX,csY,srg,tic,mit,gbp
                        datakey='A_'+'_'.join(map(str, write_to_dic))
                        #print('datakey',datakey)
                        energies={}
                        areas={}
                        latencies={}
                        if checkIfallDictExist(dnn_names,file_store_paths, datakey)==True:
                            for dnn_name in dnn_names:
                                dict_load=load_dict_from_json(file_store_paths[dnn_name])
                                #if search_instance_in_dict(dict_load, datakey)==True:
                                dataForScore=dict_load[datakey] # dictionary with data
                                total_latency=dataForScore['l']
                                total_energy=dataForScore['e']
                                total_area=dataForScore['a']
                                max_percent_utilization=dataForScore['u']
                                avg_sum_tops_per_mm2=dataForScore['a_tpm']
                                avg_sum_tops_per_w=dataForScore['a_tpw']
                                max_tops_per_w=dataForScore['m_tpw']
                                max_tops_per_mm2=dataForScore['m_tpm']
                                ##???: correct?
                                
                                energies[dnn_name]=total_energy
                                areas[dnn_name]=total_area
                                latencies[dnn_name]=total_latency
                        ############################
                        else:
                            for dnn_name in dnn_names:
                                #check=checkSizeParallel(x[i],dnn_name,layers_all[dnn_name],inp_bits,n_jobs)
                                check=True
                                if check==False:
                                    #print('Runtime error --> mutation produced too small system--> no evaluation')
                                    #energies={}
                                    #areas={}
                                    #latencies={}
                                    for dnn_name in dnn_names:
                                        energies[dnn_name]=1e25
                                        areas[dnn_name]=1e25
                                        latencies[dnn_name]=1e25
                                    break
                            if check==True: #if we reach the last check and it's true
                                #dictRes=getMetricsX(new_x[i],dnn_names)
                                
                                dictRes=getMetricsParallel(new_x[i],dnn_names,num_mapping,n_jobs)
                                #energies={}
                                #areas={}
                                #latencies={}
                                dictts={}
                                #constrGood=True
                                for dnn_name in dnn_names:
                                    #total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2=dictRes[dnn_name]
                                    total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2,tops=dictRes[dnn_name]
                                    
                                    energies[dnn_name]=total_energy
                                    areas[dnn_name]=total_area
                                    latencies[dnn_name]=total_latency
                                    #dictts[dnn_name]={'e': total_energy, 'a': total_area, 'l':total_latency,'a_tpm':avg_sum_tops_per_mm2, 'a_tpw': avg_sum_tops_per_w,'u': max_percent_utilization,'au': max_percent_utilization*total_area, 'm_tpm':max_tops_per_mm2, 'm_tpw':max_tops_per_w }
                                    
                                    dictts[dnn_name]={'e': total_energy, 'a': total_area, 'l':total_latency,'a_tpm':avg_sum_tops_per_mm2, 'a_tpw': avg_sum_tops_per_w,'u': max_percent_utilization,'au': max_percent_utilization*total_area, 'm_tpm':max_tops_per_mm2, 'm_tpw':max_tops_per_w,'tp':tops }
                                    
                                    add_new_instance_if_not_exists(file_store_paths[dnn_name], datakey, dictts[dnn_name])
                        
                                    #total_energy, total_area, total_latency, max_percent_utilization, avg_sum_tops_per_mm2, avg_sum_tops_per_w, max_tops_per_w, max_tops_per_mm2=getMetrics(new_x[i])
                                    #print('********Network:',dnn_name)
                                    #print('total_energy in J:',total_energy)
                                    #print('total_area in cm2:',total_area)
                                    #print('total_latency in s:',total_latency)
                                    #print('avg_sum_tops_per_mm2:',avg_sum_tops_per_mm2)
                                    #print('avg_sum_tops_per_w:',avg_sum_tops_per_w)
                                    #print('max_percent_utilization:',max_percent_utilization)
                                    #print('area_utilized in cm2:',max_percent_utilization*total_area)
                                    #print('max_tops_per_w:',max_tops_per_w)
                                    #print('max_tops_per_mm2:',max_tops_per_mm2)
                                areaslistconst=[]
                                for dnn_name in dnn_names:
                                    areaslistconst.append(areas[dnn_name])
                                all_good = all(x <= areaConstrcm2 for x in areaslistconst)
                                if all_good==True:
                                    for dnn_name in dnn_names:
                                        add_new_instance_if_not_exists(file_store_pathsOnlyTrue[dnn_name], datakey, dictts[dnn_name])
                                
                        latlist=list(latencies.values())
                        englist=list(energies.values())
                        arlist=list(areas.values())
                        if all(x == arlist[0] for x in arlist)==False:
                            print('not all areas are the same --> Check!')
                        areaforscore=np.mean(arlist)
                        score=objectiveMAX(latlist, englist, areaforscore, coeff_lat, coeff_en, coeff_ar,obj_type)    
                    #score=0.5

                    #### THE REST FINISHED HERE!!!!! continue:
                    
                    except RuntimeError:
                        #vl,bps,bl,csX,csY,srg,tic,mit,gbp=new_x[i]
                        vl,bl,csX,csY,srg,tic,mit,gbp=new_x[i]
                        bps=1
                        vl,bl =[float(x) for x in [vl,bl]]
                        write_to_dic=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        bps,csX,csY,srg,tic,mit,gbp=[int(x) for x in [bps,csX,csY,srg,tic,mit,gbp]]
                        write_to_dic=vl,bps,bl,csX,csY,srg,tic,mit,gbp
                        datakey='A_'+'_'.join(map(str, write_to_dic))
                        #print('datakey',datakey)
                        score=1e25 #setting for very large score (just in case)
                        areaforscore=1e25
                        #print('Runtime error --> mutation produced too small system')
                        continue
                    
                    #areaConstrcm2
                    new_f[i]=score
                    lstcons=[]
                    for dnn_name in dnn_names:
                        #total_requiredParallel(x[i],dnn_name,layers_all[dnn_name],inp_bits)-actual_present(x[i],dnn_name,layers_all[dnn_name],inp_bits)
                        lstcons.append(total_requiredParallel(x[i],dnn_name,layers_all[dnn_name],inp_bits,n_jobs)-actual_present(x[i],dnn_name,layers_all[dnn_name],inp_bits))
                    #print(lstcons.sha)
                    #print(lstcons)
                    #all_negative = all(x < 0 for x in lstcons)
                    #if all_negative==True:
                    #    new_g[i][0]=-1
                    #else: new_g[i][0]=1
                    new_g[i][:9]=lstcons
                    new_g[i][9]=areaforscore-areaConstrcm2
                    #print('np.column_stack(lstcons)')
                    #print(np.column_stack(lstcons))
                    #print('new_g')
                    #print(new_g)
                    #print('new_gi')
                    #print(new_g[i])
                    #print(type(new_g))
                    #print(type(new_g[i]))
                    #print(new_g.shape)
                    #new_g[i]=np.column_stack(lstcons)
                    #new_g[i]=np.column_stack(lstcons)[0,0]
                    #print()
                    #print(new_f[i])
                    #print(new_f)
                #new_g[i]=total_required(x[i],dnn_name,layers,inp_bits)-actual_present(x[i],dnn_name,layers,inp_bits)
                #out["X"]= newxxx
                
                #print(new_f)
                #print(new_g)
                # Open the file and read the existing data
                #with open(fileStat, 'r') as file1:
                #    dataf = [json.load(file1)]
                #dataf.extend(list(new_f))
                #with open(fileStat, 'w') as file1:
                #    json.dump(dataf, file1)
                
                #print(new_f)    
                filtered_data = new_f[new_f < 1e15]
                #print(filtered_data)
                remove_zeros = lambda lst: [x for x in lst if x != 0]
                filtered_data = remove_zeros(filtered_data)
                #print(filtered_data)
                # Calculate mean and standard deviation
                if len(filtered_data)==0:
                    mean_value=1e25
                    std_deviation =1e25
                    best = 1e25
                else:
                    mean_value = np.mean(filtered_data)
                    std_deviation = np.std(filtered_data)
                    best = min(filtered_data)
                
                #print(filtered_data)
                #print(mean_value)
                #print(std_deviation)
                #print(best)
                
                fmeanAndStd=[mean_value,std_deviation,best]
                #print(fmeanAndStd)
                with open(fileMeanStat, 'r') as file2:
                    datafm = json.load(file2)
                datafm.extend(list(list(fmeanAndStd)))
                with open(fileMeanStat, 'w') as file2:
                    json.dump(datafm, file2)
                    
                
                    
                
                out["G"] = new_g
                out["F"] = new_f
                #print('checking out["X"]')
                #print(out["X"])
        
        print('A: voltage, bits_per_cell(???), base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth')

        # initialize the thread pool and create the runner
        #n_threads = 16
        #pool = ThreadPool(n_threads)
        #runner = StarmapParallelization(pool.starmap)
        
        problem = MyProblemParallel()
        #problem = MyProblem()
        #problem = MyProblem(elementwise_runner=runner)
        start_time_search = time.time()
        method = GA(pop_size=population_size1, #20
                    #sampling=MySamplingSeveral(),
                    #sampling=MySamplingSeveralParallel(),
                    #sampling=IntegerRandomSampling(),
                    sampling=MySamplingSeveralParallelHammingDistWithEval(),
                    crossover=SBX(prob=prob_cross1, eta=eta_cross1, vtype=float, repair=RoundingRepair()),
                    mutation=PM(prob=prob_mute1, eta=eta_mute1, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                    )
        
        res = minimize(problem,
                       method,
                       termination=('n_gen', num_generations1), #20
                       seed=seed_alg, #was set to 1 to ensure repredusability
                       save_history=True,
                       verbose= True
                       )
        end_time_search = time.time()
        #print("Best solution found: %s" % res.X)
        #print('check1: ',C[res.X[0]])
        #print('check1: ',Y[res.X[1]])
        #print('check: ', res.X*[3,1])
        
        #print('voltages: ', voltages[res.X[0]])
        #print('bits_per_cell: ', bits_per_cell[res.X[1]])
        #print('base_latency: ', base_latency[res.X[2]])
        #print('crossbar_sizeX: ', crossbar_sizeX[res.X[3]])
        #print('crossbar_sizeY: ', crossbar_sizeY[res.X[4]])
        #print('shared_router_groupSize: ', shared_router_groupSize[res.X[5]])
        #print('tiles_in_chip: ', tiles_in_chip[res.X[6]])
        #print('macros_in_tile: ', macros_in_tile[res.X[7]])
        #print('glb_buffer_depth: ', glb_buffer_depth[res.X[8]])
        #print("Function value: %s" % res.F)
       # print("Function value: %s" % res.G)
        #print("Constraint violation: %s" % res.CV)
        
        #score_check(dic_scores, crossbar_size[res.X[0]], tile_multiplier[res.X[1]],tiles_per_chiplet[res.X[2]], noc_bitwidth[res.X[3]], nop_bitwidth[res.X[4]], res.F)
        
        #pop=res.pop
        #pop=res.history
        #print(dir(res.history[0]))
        #print('check inequality',checkSize(res.X,dnn_name,layers,inp_bits))
        #for ipop in range(len(pop.get("X"))):
        #    print(checkSize(pop.get("X")[ipop],dnn_name,layers,inp_bits))
        #print(pop.get("X"))
        #print(pop.get("F"))
        print('time in s:', end_time_search-start_time_search)

        problem = MyProblemParallel()
        #problem = MyProblem()
        #problem = MyProblem(elementwise_runner=runner)
        #start_time_search = time.time()
        method = GA(pop_size=population_size2, #20
                    #sampling=MySamplingSeveral(),
                    sampling=MySamplingSeveralParallelPhaseTwo(),
                    #sampling=MySamplingSeveralParallel(),
                    crossover=SBX(prob=prob_cross2, eta=eta_cross2, vtype=float, repair=RoundingRepair()),
                    mutation=PM(prob=prob_mute2, eta=eta_mute2, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                    )
        
        res = minimize(problem,
                       method,
                       termination=('n_gen', num_generations2), #20
                       seed=seed_alg, #was set to 1 to ensure repredusability
                       save_history=True,
                       verbose= True
                       )
        end_time_search = time.time()

        print('Phase 2 time in s:', end_time_search-start_time_search)

        #print('time in s:', end_time_search-start_time_search)

        problem = MyProblemParallel()
        #problem = MyProblem()
        #problem = MyProblem(elementwise_runner=runner)
        #start_time_search = time.time()
        method = GA(pop_size=population_size3, #20
                    #sampling=MySamplingSeveral(),
                    sampling=MySamplingSeveralParallelPhaseTwo(),
                    #sampling=MySamplingSeveralParallel(),
                    crossover=SBX(prob=prob_cross3, eta=eta_cross3, vtype=float, repair=RoundingRepair()),
                    mutation=PM(prob=prob_mute3, eta=eta_mute3, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                    )
        
        res = minimize(problem,
                       method,
                       termination=('n_gen', num_generations3), #20
                       seed=seed_alg, #was set to 1 to ensure repredusability
                       save_history=True,
                       verbose= True
                       )
        end_time_search = time.time()

        print('Phase 3 time in s:', end_time_search-start_time_search)

        print('time in s:', end_time_search-start_time_search)

        problem = MyProblemParallel()
        #problem = MyProblem()
        #problem = MyProblem(elementwise_runner=runner)
        #start_time_search = time.time()
        method = GA(pop_size=population_size4, #20
                    #sampling=MySamplingSeveral(),
                    sampling=MySamplingSeveralParallelPhaseTwo(),
                    #sampling=MySamplingSeveralParallel(),
                    crossover=SBX(prob=prob_cross4, eta=eta_cross4, vtype=float, repair=RoundingRepair()),
                    mutation=PM(prob=prob_mute4, eta=eta_mute4, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                    )
        
        res = minimize(problem,
                       method,
                       termination=('n_gen', num_generations4), #20
                       seed=seed_alg, #was set to 1 to ensure repredusability
                       save_history=True,
                       verbose= True
                       )
        end_time_search = time.time()

        print('Total time in s:', end_time_search-start_time_search)



