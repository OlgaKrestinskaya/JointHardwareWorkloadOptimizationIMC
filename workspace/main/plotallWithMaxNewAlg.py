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
#from _import_scripts import scripts
#from scripts.notebook_utils import *
import csv
#from scripts import *
import re
import numpy as np
# Example dictionary with nested dictionaries (attributes)
import re
import matplotlib.pyplot as plt
from HWC_RunSingle import *
coeff_lat=1.
coeff_en=1.
coeff_ar=1.

def objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type,cost=1.):
    if obj_type=='l':
        obj=max(latencys)
    elif obj_type=='e':
        obj=max(energys)
    elif obj_type=='a':
        obj=area
    elif obj_type=='el':
        obj=max(energys)*max(latencys)
    elif obj_type=='ea':
        obj=max(energys)*area
    elif obj_type=='la':
        obj=area*max(latencys)
    elif obj_type=='ela':
        obj=max(energys)*max(latencys)*area
    elif obj_type=='ela_mean':
        obj=(sum(energys)/len(energys))*(sum(latencys)/len(latencys))*area
    #return math.log(pow(latency, coeff_lat)*pow(energy, coeff_en)*pow(area, coeff_ar))
    #obj=max(energys)*max(latencys)*area
    #obj=max(latencys)
    elif obj_type=='ela_cost':
        obj=max(energys)*max(latencys)*area*cost
    elif obj_type=='el_cost':
        obj=max(energys)*max(latencys)*cost
    #elif obj_type=='ela_mean':
    #    obj=(sum(energys)/len(energys))*(sum(latencys)/len(latencys))*area
    elif obj_type == 'ela_all':
        obj = area
        for e1 in energys:
            obj *= e1
        for l1 in latencys:
            obj *= l1
    return obj

def objectiveMAXacc(latencys, energys, area,accuracies, coeff_lat, coeff_en, coeff_ar,obj_type,cost=1.):
    if obj_type=='l':
        obj=max(latencys)
    elif obj_type=='e':
        obj=max(energys)
    elif obj_type=='a':
        obj=area
    elif obj_type=='el':
        obj=max(energys)*max(latencys)
    elif obj_type=='ea':
        obj=max(energys)*area
    elif obj_type=='la':
        obj=area*max(latencys)
    elif obj_type=='ela':
        obj=max(energys)*max(latencys)*area
    elif obj_type=='ela_acc4':
        acc=(accuracies[0]/100)*(accuracies[1]/100)*(accuracies[2]/100)*(accuracies[3]/100)
        obj=max(energys)*max(latencys)*area/acc
    elif obj_type=='ela_acc4test':
        if (accuracies[0] < 87.) or (accuracies[1] < 93.) or (accuracies[2] < 89.) or (accuracies[3] < 63.):
            acc = 10e-25
        else:
            acc=(accuracies[0]/100)*(accuracies[1]/100)*(accuracies[2]/100)*(accuracies[3]/100)
        obj=max(energys)*max(latencys)*area/acc
    elif obj_type=='ela_mean':
        obj=(sum(energys)/len(energys))*(sum(latencys)/len(latencys))*area
    #return math.log(pow(latency, coeff_lat)*pow(energy, coeff_en)*pow(area, coeff_ar))
    #obj=max(energys)*max(latencys)*area
    #obj=max(latencys)
    elif obj_type=='ela_cost':
        obj=max(energys)*max(latencys)*area*cost
    elif obj_type=='el_cost':
        obj=max(energys)*max(latencys)*cost
    return obj
def objective(latency, energy, area, coeff_lat, coeff_en, coeff_ar,typeObj,cost=1.):
    if typeObj=='e':
        obj=energy
    elif typeObj=='l':
        obj=latency
    elif typeObj=='a':
        obj=area
    elif typeObj=='ela':
        obj=latency*energy*area
    elif typeObj=='el': 
        obj=latency*energy
    elif typeObj=='ea':
        obj=energy*area
    elif typeObj=='la':
        obj=area*latency
    elif typeObj=='ela_mean':
        obj=latency*energy*area
    elif typeObj=='ela_cost':
        obj=energy*latency*area*cost
    elif typeObj=='el_cost':
        obj=energy*latency*cost
    elif typeObj == 'ela_all':
        obj=latency*energy*area
    #return math.log(pow(latency, coeff_lat)*pow(energy, coeff_en)*pow(area, coeff_ar))
    return obj


def scatterDRAM(folder, obj_type, constr_type, tests,num_to_plot, num_iterations_to_run=1, num_mapping=None):
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'

    dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    
    pltcase={}
    data={}
    
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        data[dnn]=load_dict_from_json(path)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
    scores={}
    edap={}
    costs4plot={}

    eng_max={}
    lat_max={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        #costs=[data[dnn][keys]['cost'] for dnn in dnn_names]
        area=max(areas)
        #cost=max(costs)
        eng_max[keys]= max(energys)*1e3
        lat_max[keys]= max(latencys)*1e3
        scores[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type)
        edap[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,'ela')*1e3*1e9*1e2*1e-6
        # this is edap in mJ*ms*mm2
        #costs4plot[keys]=cost
        
    if num_to_plot == 'all' or num_to_plot > len(keys_list):
        num_to_plot = len(keys_list)
    
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:num_to_plot]

    # Define technology nodes
    DRAMS = [0,1,2,3,4] #= [7, 10, 14, 22, 32, 45, 65, 90]
    colors = plt.cm.viridis(np.linspace(0, 1, len(DRAMS)))  # Assign colors to technology nodes

    # Define distinct colors manually for each technology node
    tech_colors = {
        0: "#04FBD2",  # Blue
        1: "#BBA7FF", # Orange
        2: "#FFE208", # Green
        3: "#F38005", # Red
        4: "#08A8FF", # Purple
        #45: "#39DE2A", # Brown
        #65: "#532EB0", # Pink
        #90: "#098E87", # Gray
    }

    
    # Extract EDAP and Cost for top models
    eng_plot = []
    lat_plot = []
    dram_labels = []
    
    for key, _ in top_10:  # Iterate over sorted entries
        dram_node = int(key.split("_")[-1])  # Extract technology node
        if dram_node in DRAMS:
            eng_plot.append(eng_max[key])  # Extract EDAP value
            lat_plot.append(lat_max[key])  # Extract cost value
            dram_labels.append(dram_node)
    
    # Assign colors based on technology nodes
    #color_map = {tech: colors[i] for i, tech in enumerate(technologies)}
    #point_colors = [color_map[tech] for tech in tech_labels]

    # Assign colors based on technology nodes
    point_colors = [tech_colors[tech] for tech in dram_labels]


    # Step 1: Sort architectures by Cost (ascending)
    sorted_indices = np.argsort(lat_plot)  # Get indices that would sort by cost
    
    # Step 2: Iterate through sorted architectures to find Pareto-optimal points
    pareto_lat= []
    pareto_eng = []
    pareto_keys = []
    
    current_min_dap = float("inf")  # Start with a very high DAP value
    
    for i in sorted_indices:
        if eng_plot[i] < current_min_dap:  # Check if the architecture is non-dominated
            pareto_lat.append(lat_plot[i])
            pareto_eng.append(eng_plot[i])
            pareto_keys.append(top_10[i][0])  # Store corresponding architecture key
            current_min_dap = eng_plot[i]  # Update minimum DAP
    
    # Step 3: Plot original scatter plot
    fig=plt.figure(figsize=(8, 6))
# Step 4: Plot Pareto front
    #plt.plot(pareto_costs, pareto_daps, color="red", linestyle="-", marker="o", markersize=6, alpha=0.8, label="Pareto Front")
    
    scatter = plt.scatter(lat_plot, eng_plot, c=point_colors, alpha=0.6, edgecolors="black", label="All Architectures", zorder=1)
    plt.plot(pareto_lat, pareto_eng, color="red", linestyle="-",linewidth = 1, alpha=0.6, label="Pareto Front", zorder=2)

   

    # Add legend for technology nodes
    legend_labels = {tech: plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tech_colors[tech], markersize=8) for tech in tech_colors}
    legend_labels["Pareto Front"] = plt.Line2D([0], [0], color="red", alpha=0.6, label="Pareto Front")
    
    #legend_labels["Pareto Front"] = plt.Line2D([0], [0], marker='o', color="red", markersize=8, alpha=0.6, label="Pareto Front")
    plt.legend(legend_labels.values(), legend_labels.keys(), title="DRAM")
    
    plt.xlabel("Latency (ms)")
    plt.ylabel("Energy (mJ)")
    plt.title("Pareto Front of Architectures (Energy vs Latency)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()
    
    # Display Pareto-optimal architectures
    print(pareto_keys)


def scatterCOST(folder, obj_type, constr_type, tests,num_to_plot, num_iterations_to_run=1, num_mapping=None):
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'

    dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    
    pltcase={}
    data={}
    
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        data[dnn]=load_dict_from_json(path)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
    scores={}
    edap={}
    costs4plot={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        costs=[data[dnn][keys]['cost'] for dnn in dnn_names]
        area=max(areas)
        cost=max(costs)
        scores[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type,cost)
        edap[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,'ela',cost)*1e3*1e9*1e2*1e-6
        # this is edap in mJ*ms*mm2
        costs4plot[keys]=cost
        
    if num_to_plot == 'all' or num_to_plot > len(keys_list):
        num_to_plot = len(keys_list)
    
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:num_to_plot]

    # Define technology nodes
    technologies = [7, 10, 14, 22, 32, 45, 65, 90]
    colors = plt.cm.viridis(np.linspace(0, 1, len(technologies)))  # Assign colors to technology nodes

    # Define distinct colors manually for each technology node
    tech_colors = {
        7: "#04FBD2",  # Blue
        10: "#BBA7FF", # Orange
        14: "#FFE208", # Green
        22: "#F38005", # Red
        32: "#08A8FF", # Purple
        45: "#39DE2A", # Brown
        65: "#532EB0", # Pink
        90: "#098E87", # Gray
    }

    
    # Extract EDAP and Cost for top models
    edap_plot = []
    cost_plot = []
    tech_labels = []
    
    for key, _ in top_10:  # Iterate over sorted entries
        tech_node = int(key.split("_")[-1])  # Extract technology node
        if tech_node in technologies:
            edap_plot.append(edap[key])  # Extract EDAP value
            cost_plot.append(costs4plot[key])  # Extract cost value
            tech_labels.append(tech_node)
    
    # Assign colors based on technology nodes
    #color_map = {tech: colors[i] for i, tech in enumerate(technologies)}
    #point_colors = [color_map[tech] for tech in tech_labels]

    # Assign colors based on technology nodes
    point_colors = [tech_colors[tech] for tech in tech_labels]


    # Step 1: Sort architectures by Cost (ascending)
    sorted_indices = np.argsort(cost_plot)  # Get indices that would sort by cost
    
    # Step 2: Iterate through sorted architectures to find Pareto-optimal points
    pareto_costs = []
    pareto_daps = []
    pareto_keys = []
    
    current_min_dap = float("inf")  # Start with a very high DAP value
    
    for i in sorted_indices:
        if edap_plot[i] < current_min_dap:  # Check if the architecture is non-dominated
            pareto_costs.append(cost_plot[i])
            pareto_daps.append(edap_plot[i])
            pareto_keys.append(top_10[i][0])  # Store corresponding architecture key
            current_min_dap = edap_plot[i]  # Update minimum DAP
    
    # Step 3: Plot original scatter plot
    fig=plt.figure(figsize=(8, 6))
# Step 4: Plot Pareto front
    #plt.plot(pareto_costs, pareto_daps, color="red", linestyle="-", marker="o", markersize=6, alpha=0.8, label="Pareto Front")
    
    scatter = plt.scatter(cost_plot, edap_plot, c=point_colors, alpha=0.6, edgecolors="black", label="All Architectures", zorder=1)
    plt.plot(pareto_costs, pareto_daps, color="red", linestyle="-",linewidth = 1, alpha=0.6, label="Pareto Front", zorder=2)

   

    # Add legend for technology nodes
    legend_labels = {tech: plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tech_colors[tech], markersize=8) for tech in tech_colors}
    legend_labels["Pareto Front"] = plt.Line2D([0], [0], color="red", alpha=0.6, label="Pareto Front")
    #legend_labels["Pareto Front"] = plt.Line2D([0], [0], marker='o', color="red", markersize=8, alpha=0.6, label="Pareto Front")
    #plt.legend(legend_labels.values(), legend_labels.keys(), title="Technology Node (nm)")
    
    plt.xlabel("Cost")
    plt.ylabel("EDAP (mJ * ns * mm²)")
    plt.title("Pareto Front of Architectures (EDAP vs Cost)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()
    
    # Display Pareto-optimal architectures
    print(pareto_keys)
    
    #figname='results/resHWC/figures/pareto.png'
    #fig.savefig(figname,dpi=2000)



    # Create scatter plot with DAP on y-axis and Cost on x-axis
    #plt.figure(figsize=(8, 6))
    #scatter = plt.scatter(cost_plot, edap_plot, c=point_colors, alpha=0.75, edgecolors="black")
    
    # Add legend for technology nodes
    #legend_labels = {tech: plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tech_colors[tech], markersize=8) for tech in technologies}
    #plt.legend(legend_labels.values(), legend_labels.keys(), title="Technology Node (nm)")
    
    #plt.xlabel("Cost")
    #plt.ylabel("EDAP (mJ * ns * mm²)")
    #plt.title("EDAP vs Cost for Different Technology Nodes")
    #plt.xscale("log")  # Log scale for better visualization
    #plt.yscale("log")
    #plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    #plt.show()


def display_resultsCOST(folder, obj_type, constr_type, tests, num_iterations_to_run=1, num_mapping=None):

    #folder='res/new/results_part1/unconstraint/joint/joint_L_noconstr/'
    #folderSep='res/new/results_part1/unconstraint/separate/separate_L_noconstr/'
    #folder='res/resnew/joint_EL_noconstr/'
    #obj_type='l'
    #constr_type='c'
    #tests=0 #this is for a single case!
    
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'
    
    #folder='res/resnew/joint_A_nonconstr/'
    #folderSep='res/to_plot/first/allseparateRes/'
    dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    #testnum=8
    
    #pltcases={}
    #for tests in range(testnum):
    pltcase={}
    #print('')
    #print('')
    #print('Test',tests)
    #print('A: voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth')
    data={}
    
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        data[dnn]=load_dict_from_json(path)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
    scores={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        costs=[data[dnn][keys]['cost'] for dnn in dnn_names]
        area=max(areas)
        cost=max(costs)
        scores[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type,cost)
        
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:20]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_joint=top_10[0][0]
    reference_bestMaxedScore =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0
    
    bestMetricesJoint={}
    bestJointReferenceScore={}
    for dnn in dnn_names:
        bestMetricesJoint[dnn]=data[dnn][best_joint]
        bestJointReferenceScore[dnn]=objective(data[dnn][best_joint]['l'], data[dnn][best_joint]['e'], data[dnn][best_joint]['a'], coeff_lat, coeff_en, coeff_ar,obj_type,data[dnn][best_joint]['cost'])
    top10_keys=[items[0] for items in top_10]
    print('TOP 10:')
    print(top10_keys)
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    #best_scoresForEach={}
    #scores_separateJoint={}
    #for dnn in dnn_names:
        #reference_best=bestSeparateReferenceScore[dnn]
        #scores2={}
        #for keys in top10_keys:
            #scores2[keys]=objective(data[dnn][keys]['l'], data[dnn][keys]['e'], data[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        #values_list = [scores2[key] for key in top10_keys]
        #print(values_list)
        #print('dnn:', dnn)
        #print('best_Joint key:',top10_keys[0],'best_Joint value:',values_list[0])
        #best_arch_score=scores2[best_joint]
        #normalized_values = [(x - normmin) / (reference_best - normmin) for x in values_list]
        #scores_separateJoint[dnn]=normalized_values
        #best_scoresForEach[dnn]=(best_arch_score-normmin)/(reference_best - normmin)
        #print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEach[dnn])
        #diffr=best_scoresForEach[dnn]-norm_score_bestSepForRef[dnn]
        #diffrPers=diffr*100
        #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
        #allSeparateScores[dnn]=normalized_joint
        #percDiff[dnn]=diffrPers

    #print('')
    #print('Percentage difference in score (negative = joint is better)')
    #print(percDiff)
    #print('')
    #print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth','technology')
    parts = best_joint.split('_')
    print('parts', parts)
    # Regular expression to match both integers and floats
    number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbers = [part for part in parts if number_pattern.match(part)]
    numbers[0]=numbers[0]+'V'
    numbers[1]=numbers[1]+'b'
    numbers[2]=numbers[2]+'s'
    numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbers)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJoint[dnn]['e']=bestMetricesJoint[dnn]['e']*1e3 #in mJ
        bestMetricesJoint[dnn]['l']=bestMetricesJoint[dnn]['l']*1e9 #in ns
        bestMetricesJoint[dnn]['a']=bestMetricesJoint[dnn]['a']*1e2 #in mm2
        bestMetricesJoint[dnn]['cost']=bestMetricesJoint[dnn]['cost']
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJoint[dnn]['e'],'mJ,  ', 'L =',bestMetricesJoint[dnn]['l'],'ns,  ', 'A =',bestMetricesJoint[dnn]['a'],'mm2', ', COST=',bestMetricesJoint[dnn]['cost'])
        print('TOPS =',bestMetricesJoint[dnn]['tp'], ',  TOPS/W =',bestMetricesJoint[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJoint[dnn]['a_tpm'])
    

def display_resultsSINGLEforSEED_specific(folder, obj_type, constr_type, tests,dnn_names, num_iterations_to_run=1, num_mapping=None):

    #folder='res/new/results_part1/unconstraint/joint/joint_L_noconstr/'
    #folderSep='res/new/results_part1/unconstraint/separate/separate_L_noconstr/'
    #folder='res/resnew/joint_EL_noconstr/'
    #obj_type='l'
    #constr_type='c'
    #tests=0 #this is for a single case!
    
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'
    
    #folder='res/resnew/joint_A_nonconstr/'
    #folderSep='res/to_plot/first/allseparateRes/'
    #dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    #testnum=8
    
    #pltcases={}
    #for tests in range(testnum):
    pltcase={}
    #print('')
    #print('')
    #print('Test',tests)
    #print('A: voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth')
    data={}
    
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        
        data[dnn]=load_dict_from_json(path)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
    scores={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        #costs=[data[dnn][keys]['cost'] for dnn in dnn_names]
        area=max(areas)
        # already in mJ*ms*mm2
        scores[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type)*1e3*1e9*1e2*1e-6
        
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:1]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_joint=top_10[0][0]
    reference_bestMaxedScore =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0
    
    bestMetricesJoint={}
    bestJointReferenceScore={}
    for dnn in dnn_names:
        bestMetricesJoint[dnn]=data[dnn][best_joint]
        bestJointReferenceScore[dnn]=objective(data[dnn][best_joint]['l'], data[dnn][best_joint]['e'], data[dnn][best_joint]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keys=[items[0] for items in top_10]
    
    parts = best_joint.split('_')
    #print('parts', parts)
    # Regular expression to match both integers and floats
    number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbers = [part for part in parts if number_pattern.match(part)]
    numbers[0]=numbers[0]+'V'
    numbers[1]=numbers[1]+'b'
    numbers[2]=numbers[2]+'s'
    numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    #print('Best joint architectures parameters',numbers)
    
    #print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJoint[dnn]['e']=bestMetricesJoint[dnn]['e']*1e3 #in mJ
        bestMetricesJoint[dnn]['l']=bestMetricesJoint[dnn]['l']*1e3 #in ms
        bestMetricesJoint[dnn]['a']=bestMetricesJoint[dnn]['a']*1e2 #in mm2
        
        #bestMetricesJoint[dnn]['cost']=bestMetricesJoint[dnn]['cost']
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJoint[dnn]['e'],'mJ,  ', 'L =',bestMetricesJoint[dnn]['l'],'ms,  ', 'A =',bestMetricesJoint[dnn]['a'],'mm2')
        print('EDAP (mJ*ms*mm2): ', bestMetricesJoint[dnn]['e']*bestMetricesJoint[dnn]['l']*bestMetricesJoint[dnn]['a'])
        #print('TOPS =',bestMetricesJoint[dnn]['tp'], ',  TOPS/W =',bestMetricesJoint[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJoint[dnn]['a_tpm'])
        print(bestMetricesJoint[dnn])
    return (numbers,reference_bestMaxedScore)


def display_resultsTop10forSEED_specific(folder, obj_type, constr_type, tests,dnn_names, num_iterations_to_run=1, num_mapping=None):

    #folder='res/new/results_part1/unconstraint/joint/joint_L_noconstr/'
    #folderSep='res/new/results_part1/unconstraint/separate/separate_L_noconstr/'
    #folder='res/resnew/joint_EL_noconstr/'
    #obj_type='l'
    #constr_type='c'
    #tests=0 #this is for a single case!
    
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'
    
    #folder='res/resnew/joint_A_nonconstr/'
    #folderSep='res/to_plot/first/allseparateRes/'
    #dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    #testnum=8
    
    #pltcases={}
    #for tests in range(testnum):
    pltcase={}
    #print('')
    #print('')
    #print('Test',tests)
    #print('A: voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth')
    data={}
    
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        
        data[dnn]=load_dict_from_json(path)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
    scores={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        #costs=[data[dnn][keys]['cost'] for dnn in dnn_names]
        area=max(areas)
        # already in mJ*ms*mm2
        scores[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type)*1e3*1e9*1e2*1e-6
        
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:10]
    print(top_10)
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_joint=top_10[0][0]
    reference_bestMaxedScore =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0
    
    bestMetricesJoint={}
    bestJointReferenceScore={}
    for dnn in dnn_names:
        bestMetricesJoint[dnn]=data[dnn][best_joint]
        bestJointReferenceScore[dnn]=objective(data[dnn][best_joint]['l'], data[dnn][best_joint]['e'], data[dnn][best_joint]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keys=[items[0] for items in top_10]
    
    parts = best_joint.split('_')
    #print('parts', parts)
    # Regular expression to match both integers and floats
    number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbers = [part for part in parts if number_pattern.match(part)]
    numbers[0]=numbers[0]+'V'
    numbers[1]=numbers[1]+'b'
    numbers[2]=numbers[2]+'s'
    numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    #print('Best joint architectures parameters',numbers)
    
    #print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJoint[dnn]['e']=bestMetricesJoint[dnn]['e']*1e3 #in mJ
        bestMetricesJoint[dnn]['l']=bestMetricesJoint[dnn]['l']*1e3 #in ms
        bestMetricesJoint[dnn]['a']=bestMetricesJoint[dnn]['a']*1e2 #in mm2
        
        #bestMetricesJoint[dnn]['cost']=bestMetricesJoint[dnn]['cost']
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJoint[dnn]['e'],'mJ,  ', 'L =',bestMetricesJoint[dnn]['l'],'ms,  ', 'A =',bestMetricesJoint[dnn]['a'],'mm2')
        #print('TOPS =',bestMetricesJoint[dnn]['tp'], ',  TOPS/W =',bestMetricesJoint[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJoint[dnn]['a_tpm'])
    return (numbers,reference_bestMaxedScore)

def display_resultsSINGLEforSEED(folder, obj_type, constr_type, tests, num_iterations_to_run=1, num_mapping=None):

    #folder='res/new/results_part1/unconstraint/joint/joint_L_noconstr/'
    #folderSep='res/new/results_part1/unconstraint/separate/separate_L_noconstr/'
    #folder='res/resnew/joint_EL_noconstr/'
    #obj_type='l'
    #constr_type='c'
    #tests=0 #this is for a single case!
    
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'
    
    #folder='res/resnew/joint_A_nonconstr/'
    #folderSep='res/to_plot/first/allseparateRes/'
    dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    #testnum=8
    
    #pltcases={}
    #for tests in range(testnum):
    pltcase={}
    #print('')
    #print('')
    #print('Test',tests)
    #print('A: voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth')
    data={}
    data_nonconstraint={}
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        path_data_nonconstraint=folder+dnn+'/test'+str(tests)+'.json'
        data[dnn]=load_dict_from_json(path)
        data_nonconstraint[dnn]=load_dict_from_json(path_data_nonconstraint)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
            keys_list_nonconstr=list(data_nonconstraint[dnn].keys())
    scores={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        #costs=[data[dnn][keys]['cost'] for dnn in dnn_names]
        area=max(areas)
        # already in mJ*ms*mm2
        scores[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type)*1e3*1e9*1e2*1e-6
        
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:1]

    
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_joint=top_10[0][0]
    reference_bestMaxedScore =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)

    num_designs = len(keys_list)
    original_index = keys_list.index(best_joint)
    print('Number of entries: ', num_designs)
    print('Index of the best design: ', original_index)
    num_designs_nonconstr = len(keys_list_nonconstr)
    original_index_nonconstr = keys_list_nonconstr.index(best_joint)
    print('Number of entries nonconstrains: ', num_designs_nonconstr)
    print('Index of the best design nonconstr: ', original_index_nonconstr)

    
    normmin=0
    
    bestMetricesJoint={}
    bestJointReferenceScore={}
    for dnn in dnn_names:
        bestMetricesJoint[dnn]=data[dnn][best_joint]
        bestJointReferenceScore[dnn]=objective(data[dnn][best_joint]['l'], data[dnn][best_joint]['e'], data[dnn][best_joint]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keys=[items[0] for items in top_10]
    
    #print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth','technology')
    parts = best_joint.split('_')
    #print('parts', parts)
    # Regular expression to match both integers and floats
    number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbers = [part for part in parts if number_pattern.match(part)]
    numbers[0]=numbers[0]+'V'
    numbers[1]=numbers[1]+'b'
    numbers[2]=numbers[2]+'s'
    numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    #print('Best joint architectures parameters',numbers)
    
    #print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJoint[dnn]['e']=bestMetricesJoint[dnn]['e']*1e3 #in mJ
        bestMetricesJoint[dnn]['l']=bestMetricesJoint[dnn]['l']*1e3 #in ms
        bestMetricesJoint[dnn]['a']=bestMetricesJoint[dnn]['a']*1e2 #in mm2
        
        #bestMetricesJoint[dnn]['cost']=bestMetricesJoint[dnn]['cost']
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJoint[dnn]['e'],'mJ,  ', 'L =',bestMetricesJoint[dnn]['l'],'ms,  ', 'A =',bestMetricesJoint[dnn]['a'],'mm2')
        print('EDAP (mJ*ms*mm2): ', bestMetricesJoint[dnn]['e']*bestMetricesJoint[dnn]['l']*bestMetricesJoint[dnn]['a'])
        print(bestMetricesJoint[dnn])
        #print('TOPS =',bestMetricesJoint[dnn]['tp'], ',  TOPS/W =',bestMetricesJoint[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJoint[dnn]['a_tpm'])
    return (numbers,reference_bestMaxedScore)




def display_resultsSINGLEforSEEDaccuracy(folder, obj_type, constr_type, tests, num_iterations_to_run=1, num_mapping=None):

    #folder='res/new/results_part1/unconstraint/joint/joint_L_noconstr/'
    #folderSep='res/new/results_part1/unconstraint/separate/separate_L_noconstr/'
    #folder='res/resnew/joint_EL_noconstr/'
    #obj_type='l'
    #constr_type='c'
    #tests=0 #this is for a single case!
    
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'
    
    #folder='res/resnew/joint_A_nonconstr/'
    #folderSep='res/to_plot/first/allseparateRes/'
    dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    #testnum=8
    
    #pltcases={}
    #for tests in range(testnum):
    pltcase={}
    #print('')
    #print('')
    #print('Test',tests)
    #print('A: voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth')
    data={}
    
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        
        data[dnn]=load_dict_from_json(path)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
    scores={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        accuracies=[data[dnn][keys]['acc'] for dnn in dnn_names]
        area=max(areas)
        
        # already in mJ*ms*mm2
        scores[keys]=objectiveMAXacc(latencys, energys, area,accuracies, coeff_lat, coeff_en, coeff_ar,obj_type)*1e3*1e9*1e2*1e-6
        
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:100]
    print(top_10)
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_joint=top_10[0][0]
    reference_bestMaxedScore =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0
    
    bestMetricesJoint={}
    #bestJointReferenceScore={}
    for dnn in dnn_names:
        bestMetricesJoint[dnn]=data[dnn][best_joint]
        #bestJointReferenceScore[dnn]=objective(data[dnn][best_joint]['l'], data[dnn][best_joint]['e'], data[dnn][best_joint]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keys=[items[0] for items in top_10]
    
    #print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth','technology')
    parts = best_joint.split('_')
    #print('parts', parts)
    # Regular expression to match both integers and floats
    number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbers = [part for part in parts if number_pattern.match(part)]
    numbers[0]=numbers[0]+'V'
    numbers[1]=numbers[1]+'b'
    numbers[2]=numbers[2]+'s'
    numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    #print('Best joint architectures parameters',numbers)
    
    #print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJoint[dnn]['e']=bestMetricesJoint[dnn]['e']*1e3 #in mJ
        bestMetricesJoint[dnn]['l']=bestMetricesJoint[dnn]['l']*1e3 #in ms
        bestMetricesJoint[dnn]['a']=bestMetricesJoint[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['acc']=bestMetricesJoint[dnn]['a']
        #bestMetricesJoint[dnn]['cost']=bestMetricesJoint[dnn]['cost']
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJoint[dnn]['e'],'mJ,  ', 'L =',bestMetricesJoint[dnn]['l'],'ms,  ', 'A =',bestMetricesJoint[dnn]['a'],'mm2',bestMetricesJoint[dnn]['acc'],'%')
        #print('TOPS =',bestMetricesJoint[dnn]['tp'], ',  TOPS/W =',bestMetricesJoint[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJoint[dnn]['a_tpm'])
    return (numbers,reference_bestMaxedScore)


def display_results(folder, folderSep,folderNew, obj_type, constr_type, tests, figname, ylimit, n_jobs, num_iterations_to_run=1, num_mapping=None):

    #folder='res/new/results_part1/unconstraint/joint/joint_L_noconstr/'
    #folderSep='res/new/results_part1/unconstraint/separate/separate_L_noconstr/'
    #folder='res/resnew/joint_EL_noconstr/'
    #obj_type='l'
    #constr_type='c'
    #tests=0 #this is for a single case!
    
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'
    
    #folder='res/resnew/joint_A_nonconstr/'
    #folderSep='res/to_plot/first/allseparateRes/'
    dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    #testnum=8
    
    #pltcases={}
    #for tests in range(testnum):
    pltcase={}
    #print('')
    #print('')
    #print('Test',tests)
    #print('A: voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth')
    data={}
    
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        data[dnn]=load_dict_from_json(path)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
    scores={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        area=max(areas)
        scores[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type)
        
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:10]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_joint=top_10[0][0]
    reference_bestMaxedScore =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0
    
    
    # HERE IS SEPARATE:
    print('')
    print('THIS IS FOR SEPARATE:')
    print('')

     #tests3=3
    dataSep={}
    keys_listSep={}
    
    for dnn in dnn_names:
        path=folderSep+dnn+tosep+str(tests)+'.json'
        #path=folderSep+dnn+'/test'+str(tests3)+'.json'
        dataSep[dnn]=load_dict_from_json(path)
        keys_listSep[dnn] = list(dataSep[dnn].keys())
    
    allSeparateScores={}
    percDiff={}
    bestMetricesSep={}
    #for dnn in dnn_names:
    bestArchSep={}
    bestSeparateReferenceScore={}
    norm_score_bestSepForRef={}
    for dnn in dnn_names:
        #reference_best=bestJointReferenceScore[dnn]
        scoresSep={}
        for keys in keys_listSep[dnn]:
            scoresSep[keys]=objective(dataSep[dnn][keys]['l'], dataSep[dnn][keys]['e'], dataSep[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar, obj_type)
        top_10Sep = sorted(scoresSep.items(), key=lambda x: x[1], reverse=False)[:10]
        print('dnn:', dnn)
        #print('best_Sep key',top_10Sep[0][0])
        print('best_Sep value',top_10Sep[0][1])
        best_Sep = top_10Sep[0][0]  # Get the first key in the sorted dictionary
        bestSeparateReferenceScore[dnn]=top_10Sep[0][1]
        reference_best=bestSeparateReferenceScore[dnn]
        
        bestArchSep[dnn]=best_Sep
        bestMetricesSep[dnn]=dataSep[dnn][best_Sep]
        score_bestSep = top_10Sep[0][1]
        norm_score_bestSep=(score_bestSep- normmin)/(reference_best - normmin)
        norm_score_bestSepForRef[dnn]=norm_score_bestSep
        diffr=norm_score_bestSep-norm_score_bestSep
        diffrPers=diffr*100
        normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10Sep]
        allSeparateScores[dnn]=normalized_joint
        print('allSeparateScores[dnn]:', allSeparateScores[dnn])
        percDiff[dnn]=diffrPers
    #print('allSeparateScores')
    #print(allSeparateScores)
    print('')
    print('Percentage difference in score (negative = joint is better): separate')
    print(percDiff)
    print('')
    print('best separate')
    for dnn in dnn_names:
        print('   ', dnn)
        parts = bestArchSep[dnn].split('_')
        # Regular expression to match both integers and floats
        number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        # Extract numbers (integers or floats) from the split parts
        numbers = [part for part in parts if number_pattern.match(part)]
        numbers[0]=numbers[0]+'V'
        numbers[1]=numbers[1]+'b'
        numbers[2]=numbers[2]+'s'
        numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
        print('Best params separate',numbers)
        
        bestMetricesSep[dnn]['e']=bestMetricesSep[dnn]['e']*1e3 #in mJ
        bestMetricesSep[dnn]['l']=bestMetricesSep[dnn]['l']*1e9 #in ns
        bestMetricesSep[dnn]['a']=bestMetricesSep[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesSep[dnn]['e'],'mJ,  ', 'L =',bestMetricesSep[dnn]['l'],'ns,  ', 'A =',bestMetricesSep[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesSep[dnn]['tp'], ',  TOPS/W =',bestMetricesSep[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesSep[dnn]['a_tpm'])



    
    print('')
    print('THIS IS FOR JOINT:')
    print('')
    #HERE IS NORMALIZATION FOR JOINT:
    
    bestMetricesJoint={}
    bestJointReferenceScore={}
    for dnn in dnn_names:
        bestMetricesJoint[dnn]=data[dnn][best_joint]
        bestJointReferenceScore[dnn]=objective(data[dnn][best_joint]['l'], data[dnn][best_joint]['e'], data[dnn][best_joint]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keys=[items[0] for items in top_10]
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    best_scoresForEach={}
    scores_separateJoint={}
    for dnn in dnn_names:
        reference_best=bestSeparateReferenceScore[dnn]
        scores2={}
        for keys in top10_keys:
            scores2[keys]=objective(data[dnn][keys]['l'], data[dnn][keys]['e'], data[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        values_list = [scores2[key] for key in top10_keys]
        #print(values_list)
        print('dnn:', dnn)
        print('best_Joint key:',top10_keys[0],'best_Joint value:',values_list[0])
        best_arch_score=scores2[best_joint]
        normalized_values = [(x - normmin) / (reference_best - normmin) for x in values_list]
        scores_separateJoint[dnn]=normalized_values
        best_scoresForEach[dnn]=(best_arch_score-normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEach[dnn])
        diffr=best_scoresForEach[dnn]-norm_score_bestSepForRef[dnn]
        diffrPers=diffr*100
        normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
        #allSeparateScores[dnn]=normalized_joint
        percDiff[dnn]=diffrPers

    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiff)
    print('')
    print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth')
    parts = best_joint.split('_')
    print('parts', parts)
    # Regular expression to match both integers and floats
    number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbers = [part for part in parts if number_pattern.match(part)]
    numbers[0]=numbers[0]+'V'
    numbers[1]=numbers[1]+'b'
    numbers[2]=numbers[2]+'s'
    numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbers)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJoint[dnn]['e']=bestMetricesJoint[dnn]['e']*1e3 #in mJ
        bestMetricesJoint[dnn]['l']=bestMetricesJoint[dnn]['l']*1e9 #in ns
        bestMetricesJoint[dnn]['a']=bestMetricesJoint[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJoint[dnn]['e'],'mJ,  ', 'L =',bestMetricesJoint[dnn]['l'],'ns,  ', 'A =',bestMetricesJoint[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesJoint[dnn]['tp'], ',  TOPS/W =',bestMetricesJoint[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJoint[dnn]['a_tpm'])
    
    
    
   
    print('')
    print('THIS IS FOR MAX:')
    print('')
    
    #FOR MAX:
    #tests3=3
    dataMax={}
    keys_listMax={}
    dnn_names2=["vgg16"]
    dnn_names3=["resnet18","alexnet","mobilenet_v3"]
    for dnn in dnn_names2:
        path=folderSep+dnn+tosep+str(tests)+'.json'
        #path=folderSep+dnn+'/test'+str(tests3)+'.json'
        dataMax[dnn]=load_dict_from_json(path)
        keys_listMax[dnn] = list(dataMax[dnn].keys())
    
    
    print('')
    allSeparateScoresMAX={}
    percDiffMAX={}
    bestMetricesMax={}
    #for dnn in dnn_names:
    bestArchMax={}    
    for dnn in dnn_names2:
        reference_best=bestSeparateReferenceScore[dnn]
        scoresMax={}
        for keys in keys_listMax[dnn]:
            scoresMax[keys]=objective(dataMax[dnn][keys]['l'], dataMax[dnn][keys]['e'], dataMax[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar, obj_type)
        top_10SepMax = sorted(scoresMax.items(), key=lambda x: x[1], reverse=False)[:10]
        print('dnn:', dnn)
        #print(top_10SepMax)
        print('key',top_10SepMax[0][0])
        print('value',top_10SepMax[0][1])
        best_Sep = top_10SepMax[0][0]  # Get the first key in the sorted dictionary
        bestArchMax[dnn]=best_Sep
        bestMetricesMax[dnn]=dataMax[dnn][best_Sep]
        score_bestSep = top_10SepMax[0][1]
        norm_score_bestSep=(score_bestSep- normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: norm_score_bestSep (normalized score to reference)', norm_score_bestSep)
        diffr=norm_score_bestSepForRef[dnn]-norm_score_bestSep
        diffrPers=diffr*100
        normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10SepMax]
        allSeparateScoresMAX[dnn]=normalized_joint
        percDiffMAX[dnn]=diffrPers

    first_elements = []
    topBestForAnalysis=[]
    for item in top_10SepMax:
        first_elements.append(item[0])
    for element in first_elements:
        parts = element.split('_')
        # Regular expression to match both integers and floats
        number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        # Extract numbers (integers or floats) from the split parts
        numbers = [part for part in parts if number_pattern.match(part)]
        #print(numbers)
        converted_values = [float(val) if 'e' in val or '.' in val else int(val) for val in numbers]
        topBestForAnalysis.append(converted_values)

    print('topBestForAnalysis',topBestForAnalysis)

    maxres=[]
    print('Analysing MAX')
    for keys in topBestForAnalysis:
        print('Evaluating', keys)
        hardware_params=keys
        dictts=runSingle(dnn_names3, num_iterations_to_run, num_mapping,n_jobs, hardware_params)
        maxres.append(dictts)
        
    #print('maxres',maxres)
    
    
    for dnn in dnn_names3:
        print('Analysing MAX', dnn)
        reference_best=bestSeparateReferenceScore[dnn]
        scoresMax=[]
        # Printing the first elements  
        for i, keys in enumerate(topBestForAnalysis):
            print('Evaluating', keys, 'i',  i)
            scoresMax.append(objective(maxres[i][dnn]['l'], maxres[i][dnn]['e'], maxres[i][dnn]['a'], coeff_lat, coeff_en, coeff_ar, obj_type))
        print('scoresMax',scoresMax)
        print('dnn:', dnn)
        #print(top_10SepMax)
        print('key',topBestForAnalysis[0])
        print('value',scoresMax[0])
        #top_10Sep = sorted(scoresMax.items(), key=lambda x: x[1], reverse=False)[:10]
        #print(top_10Sep)
        #print(top_10Sep[0])
        #print(top_10Sep[1])
        #best_Sep = top_10Sep[0][0]  # Get the first key in the sorted dictionary
        bestArchMax[dnn]=top_10SepMax[0][0]
        score_bestSep = scoresMax[0]
        norm_score_bestSep=(score_bestSep- normmin)/(reference_best - normmin)
        print('norm_score_bestSep (normalized score to reference)', norm_score_bestSep)
        diffr=norm_score_bestSep-best_scoresForEach[dnn]
        diffrPers=diffr*100
        normalized_joint = [(items - normmin) / (reference_best - normmin) for items in scoresMax]
        allSeparateScoresMAX[dnn]=normalized_joint
        percDiffMAX[dnn]=diffrPers
    
    #print('allSeparateScores')
    #print(allSeparateScores)
    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiffMAX)
    print('')
    print('best separate')
    for dnn in dnn_names2:
        print('   ', dnn)
        parts = bestArchMax[dnn].split('_')
        # Regular expression to match both integers and floats
        number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        # Extract numbers (integers or floats) from the split parts
        
        numbers = [part for part in parts if number_pattern.match(part)]
        #print(numbers)
        #converted_values = [float(val) if 'e' in val or '.' in val else int(val) for val in a]

        numbers[0]=numbers[0]+'V'
        numbers[1]=numbers[1]+'b'
        numbers[2]=numbers[2]+'s'
        numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
        print('Best params separate',numbers)
        
        bestMetricesMax[dnn]['e']=bestMetricesMax[dnn]['e']*1e3 #in mJ
        bestMetricesMax[dnn]['l']=bestMetricesMax[dnn]['l']*1e9 #in ns
        bestMetricesMax[dnn]['a']=bestMetricesMax[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesMax[dnn]['e'],'mJ,  ', 'L =',bestMetricesMax[dnn]['l'],'ns,  ', 'A =',bestMetricesMax[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesMax[dnn]['tp'], ',  TOPS/W =',bestMetricesMax[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesMax[dnn]['a_tpm'])

    # PRINT STARTS HERE
    
    #print('log')
    #print([math.log(item) for item in allSeparateScores['resnet18']])

    
    print('')
    print('THIS IS FOR JOINT NEW ALG:')
    print('')
    

    dataNEW={}
    
    for dnn in dnn_names:
        pathNEW=folderNew+dnn+'/test'+str(tests)+ending
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
        
    top_10NEW = sorted(scoresNEW.items(), key=lambda x: x[1], reverse=False)[:10]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_jointNEW=top_10NEW[0][0]
    reference_bestMaxedScoreNEW =top_10NEW[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0


    percDiffNEW={}
    
    bestMetricesJointNEW={}
    bestJointReferenceScoreNEW={}
    for dnn in dnn_names:
        bestMetricesJointNEW[dnn]=dataNEW[dnn][best_jointNEW]
        bestJointReferenceScoreNEW[dnn]=objective(dataNEW[dnn][best_jointNEW]['l'], dataNEW[dnn][best_jointNEW]['e'], dataNEW[dnn][best_jointNEW]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keysNEW=[items[0] for items in top_10NEW]
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    best_scoresForEachNEW={}
    scores_separateJointNEW={}
    for dnn in dnn_names:
        reference_best=bestSeparateReferenceScore[dnn]
        scores2NEW={}
        for keys in top10_keysNEW:
            scores2NEW[keys]=objective(dataNEW[dnn][keys]['l'], dataNEW[dnn][keys]['e'], dataNEW[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        values_listNEW = [scores2NEW[key] for key in top10_keysNEW]
        #print(values_list)
        print('dnn:', dnn)
        print('best_Joint key:',top10_keysNEW[0],'best_Joint value:',values_listNEW[0])
        best_arch_scoreNEW=scores2NEW[best_jointNEW]
        normalized_valuesNEW = [(x - normmin) / (reference_best - normmin) for x in values_listNEW]
        scores_separateJointNEW[dnn]=normalized_valuesNEW
        best_scoresForEachNEW[dnn]=(best_arch_scoreNEW-normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEachNEW[dnn])
        diffrNEW=best_scoresForEachNEW[dnn]-norm_score_bestSepForRef[dnn]
        diffrPersNEW=diffrNEW*100
        normalized_jointNEW = [(items[1] - normmin) / (reference_best - normmin) for items in top_10NEW]
        #allSeparateScores[dnn]=normalized_joint
        percDiffNEW[dnn]=diffrPersNEW

    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiffNEW)
    print('')
    print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth')
    partsNEW = best_jointNEW.split('_')
    print(partsNEW)
    # Regular expression to match both integers and floats
    number_patternNEW = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbersNEW = [part for part in partsNEW if number_patternNEW.match(part)]
    numbersNEW[0]=numbersNEW[0]+'V'
    numbersNEW[1]=numbersNEW[1]+'b'
    numbersNEW[2]=numbersNEW[2]+'s'
    numbersNEW[8]=str(round(int(numbersNEW[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbersNEW)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJointNEW[dnn]['e']=bestMetricesJointNEW[dnn]['e']*1e3 #in mJ
        bestMetricesJointNEW[dnn]['l']=bestMetricesJointNEW[dnn]['l']*1e9 #in ns
        bestMetricesJointNEW[dnn]['a']=bestMetricesJointNEW[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJointNEW[dnn]['e'],'mJ,  ', 'L =',bestMetricesJointNEW[dnn]['l'],'ns,  ', 'A =',bestMetricesJointNEW[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesJointNEW[dnn]['tp'], ',  TOPS/W =',bestMetricesJointNEW[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJointNEW[dnn]['a_tpm'])
    
    
    
    # HERE PLOTTING STARTS!!!!!
    
    # Define separate lists for x and y values where x has fewer unique elements compared to y
    x_values = [1, 2, 3,4,6,7,8,9,11,12, 13, 14, 16,17,18,19]  # Unique x values
    
    percentage=[str(abs(round(percDiff[dnn])))+'%' for dnn in dnn_names]
    y_values = [
        
        allSeparateScores[dnn_names[0]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[0]],
        scores_separateJoint[dnn_names[0]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[0]], 
        
        
        allSeparateScores[dnn_names[1]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[1]],
        scores_separateJoint[dnn_names[1]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[1]], 
        
        allSeparateScores[dnn_names[2]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[2]],
        scores_separateJoint[dnn_names[2]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[2]], 
        
        allSeparateScores[dnn_names[3]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[3]],
        scores_separateJoint[dnn_names[3]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[3]]
    ]
    
    #x_labels = ['Joint'+dnn_names[0], 'Separate'+dnn_names[0],'Joint'+dnn_names[1], 'Separate'+dnn_names[1],'Joint'+dnn_names[2], 'Separate'+dnn_names[2],'Joint'+dnn_names[3], 'Separate'+dnn_names[3]]
    x_labels = ['Separate', 'Max','Joint','JointNew', 'Separate', 'Max','Joint','JointNew', 'Separate', 'Max','Joint', 'JointNew','Separate', 'Max','Joint','JointNew']
    
    # Define custom hex colors for each group
    #hex_colors = ['#76d3d7', '#76d3d7', '#a8e436', '#a8e436', '#a3cbef','#a3cbef', '#ffe822', '#ffe822']  # Blue, Green, and Orange shades
    #edge_colors=['#55979a', '#55979a', '#658922', '#658922', '#647e95','#647e95', '#9e901a', '#9e901a']
    hex_colors = ['#4361ee','#4361ee','#4361ee','#4361ee', '#20d4f6', '#20d4f6','#20d4f6','#20d4f6','#d242ff','#d242ff', '#d242ff', '#d242ff', '#2a9d8f', '#2a9d8f', '#2a9d8f', '#2a9d8f']  # Blue, Green, and Orange shades
    edge_colors=['#32267d', '#32267d','#32267d', '#32267d','#2e7294', '#2e7294', '#2e7294','#2e7294','#470080','#470080','#470080','#470080', '#264653', '#264653', '#264653', '#264653']
    
    # Flatten the x and y lists to create a single list for plotting
    x_flat = [x for x, y_list in zip(x_values, y_values) for _ in y_list]  # Repeat each x according to the y elements
    y_flat = [y for y_list in y_values for y in y_list]  # Flatten all y values
    
    # Create the scatter plot
    
    fig1=plt.figure(figsize=(10, 6))
    
    # Plot each group with the specified hex colors and highlight the score with a star
    for i, (x, y_vals) in enumerate(zip(x_values, y_values)):
        if (i + 1) % 4 == 1:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='^',s=300, label=f'Group {x_labels[i]}')
        elif (i + 1) % 4 == 2:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='s',s=300, label=f'Group {x_labels[i]}')
        elif (i + 1) % 4 == 3:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='o',s=300, label=f'Group {x_labels[i]}')
        else:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='v',s=300, label=f'Group {x_labels[i]}')
        # Highlight one dot per x value as a star with the same hex color
        plt.scatter(x, y_vals[0], color=hex_colors[i],edgecolors='#0E3C34', marker='*', s=750)  # 's' controls the size of the star
        # Add a custom legend text directly on the graph next to the first star point of each group
        
        # Add a custom legend text directly on the graph for even-numbered groups only
        #if (i + 1) % 2 == 0:  # Check if the group is even (i+1 because index i starts from 0)
        #    plt.text(x + 0.3, y_vals[0], f'{percentage[round(np.floor(i/2))]}', fontsize=10, color=hex_colors[i], va='center')
    
    
  
    # Adding custom legends
    custom_legends = [
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[0], markersize=10, label=dnn_names[0]),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[2], markersize=10, label=dnn_names[1]),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[4], markersize=10, label=dnn_names[2]),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[6], markersize=10, label=dnn_names[3]),
        #plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#0E3C34', markersize=15, label='Best architecture')
    ]
    
    plt.xlim(0, 20)  # Start from 0 and end at 4 on x-axis
    plt.ylim(0, ylimit) 
    #plt.xticks([])
    #plt.yticks([])
    #plt.legend(handles=custom_legends)
    # Adding labels and title
    #plt.xlabel('X-axis')
    #plt.ylabel('Normalized score')
    #plt.title('Scatter Plot with Separate X and Y Lists')
    plt.xticks(ticks=x_values, labels=x_labels)
    #plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.grid(axis='x')
    # Display the plot
    plt.tight_layout()
    plt.show()
    fig1.savefig(figname,dpi=2000)


def display_results5(folder, folderSep,folderNew, folderNew2, obj_type, constr_type, tests, figname, ylimit, n_jobs, numBEST, num_iterations_to_run=1, num_mapping=None):

    #folder='res/new/results_part1/unconstraint/joint/joint_L_noconstr/'
    #folderSep='res/new/results_part1/unconstraint/separate/separate_L_noconstr/'
    #folder='res/resnew/joint_EL_noconstr/'
    #obj_type='l'
    #constr_type='c'
    #tests=0 #this is for a single case!
    
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'
    
    #folder='res/resnew/joint_A_nonconstr/'
    #folderSep='res/to_plot/first/allseparateRes/'
    dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    #testnum=8
    
    #pltcases={}
    #for tests in range(testnum):
    pltcase={}
    #print('')
    #print('')
    #print('Test',tests)
    #print('A: voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth')
    data={}
    
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        data[dnn]=load_dict_from_json(path)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
    scores={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        area=max(areas)
        scores[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type)
        
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:numBEST]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_joint=top_10[0][0]
    reference_bestMaxedScore =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0
    
    
    # HERE IS SEPARATE:
    print('')
    print('THIS IS FOR SEPARATE:')
    print('')

     #tests3=3
    dataSep={}
    keys_listSep={}
    
    for dnn in dnn_names:
        path=folderSep+dnn+tosep+str(tests)+'.json'
        #path=folderSep+dnn+'/test'+str(tests3)+'.json'
        dataSep[dnn]=load_dict_from_json(path)
        keys_listSep[dnn] = list(dataSep[dnn].keys())
    
    allSeparateScores={}
    percDiff={}
    bestMetricesSep={}
    #for dnn in dnn_names:
    bestArchSep={}
    bestSeparateReferenceScore={}
    norm_score_bestSepForRef={}
    for dnn in dnn_names:
        #reference_best=bestJointReferenceScore[dnn]
        scoresSep={}
        for keys in keys_listSep[dnn]:
            scoresSep[keys]=objective(dataSep[dnn][keys]['l'], dataSep[dnn][keys]['e'], dataSep[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar, obj_type)
        top_10Sep = sorted(scoresSep.items(), key=lambda x: x[1], reverse=False)[:numBEST]
        print('dnn:', dnn)
        #print('best_Sep key',top_10Sep[0][0])
        print('best_Sep value',top_10Sep[0][1])
        best_Sep = top_10Sep[0][0]  # Get the first key in the sorted dictionary
        bestSeparateReferenceScore[dnn]=top_10Sep[0][1]
        reference_best=bestSeparateReferenceScore[dnn]
        
        bestArchSep[dnn]=best_Sep
        bestMetricesSep[dnn]=dataSep[dnn][best_Sep]
        score_bestSep = top_10Sep[0][1]
        norm_score_bestSep=(score_bestSep- normmin)/(reference_best - normmin)
        norm_score_bestSepForRef[dnn]=norm_score_bestSep
        diffr=norm_score_bestSep-norm_score_bestSep
        diffrPers=diffr*100
        normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10Sep]
        allSeparateScores[dnn]=normalized_joint
        print('allSeparateScores[dnn]:', allSeparateScores[dnn])
        percDiff[dnn]=diffrPers
    #print('allSeparateScores')
    #print(allSeparateScores)
    print('')
    print('Percentage difference in score (negative = joint is better): separate')
    print(percDiff)
    print('')
    print('best separate')
    for dnn in dnn_names:
        print('   ', dnn)
        parts = bestArchSep[dnn].split('_')
        # Regular expression to match both integers and floats
        number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        # Extract numbers (integers or floats) from the split parts
        numbers = [part for part in parts if number_pattern.match(part)]
        numbers[0]=numbers[0]+'V'
        numbers[1]=numbers[1]+'b'
        numbers[2]=numbers[2]+'s'
        numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
        print('Best params separate',numbers)
        
        bestMetricesSep[dnn]['e']=bestMetricesSep[dnn]['e']*1e3 #in mJ
        bestMetricesSep[dnn]['l']=bestMetricesSep[dnn]['l']*1e9 #in ns
        bestMetricesSep[dnn]['a']=bestMetricesSep[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesSep[dnn]['e'],'mJ,  ', 'L =',bestMetricesSep[dnn]['l'],'ns,  ', 'A =',bestMetricesSep[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesSep[dnn]['tp'], ',  TOPS/W =',bestMetricesSep[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesSep[dnn]['a_tpm'])
        



    
    print('')
    print('THIS IS FOR JOINT:')
    print('')
    #HERE IS NORMALIZATION FOR JOINT:
    
    bestMetricesJoint={}
    bestJointReferenceScore={}
    for dnn in dnn_names:
        bestMetricesJoint[dnn]=data[dnn][best_joint]
        bestJointReferenceScore[dnn]=objective(data[dnn][best_joint]['l'], data[dnn][best_joint]['e'], data[dnn][best_joint]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keys=[items[0] for items in top_10]
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    best_scoresForEach={}
    scores_separateJoint={}
    for dnn in dnn_names:
        reference_best=bestSeparateReferenceScore[dnn]
        scores2={}
        for keys in top10_keys:
            scores2[keys]=objective(data[dnn][keys]['l'], data[dnn][keys]['e'], data[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        values_list = [scores2[key] for key in top10_keys]
        #print(values_list)
        print('dnn:', dnn)
        print('best_Joint key:',top10_keys[0],'best_Joint value:',values_list[0])
        best_arch_score=scores2[best_joint]
        normalized_values = [(x - normmin) / (reference_best - normmin) for x in values_list]
        scores_separateJoint[dnn]=normalized_values
        best_scoresForEach[dnn]=(best_arch_score-normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEach[dnn])
        diffr=best_scoresForEach[dnn]-norm_score_bestSepForRef[dnn]
        diffrPers=diffr*100
        normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
        #allSeparateScores[dnn]=normalized_joint
        percDiff[dnn]=diffrPers

    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiff)
    print('')
    print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth')
    parts = best_joint.split('_')
    print('parts', parts)
    # Regular expression to match both integers and floats
    number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbers = [part for part in parts if number_pattern.match(part)]
    numbers[0]=numbers[0]+'V'
    numbers[1]=numbers[1]+'b'
    numbers[2]=numbers[2]+'s'
    numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbers)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJoint[dnn]['e']=bestMetricesJoint[dnn]['e']*1e3 #in mJ
        bestMetricesJoint[dnn]['l']=bestMetricesJoint[dnn]['l']*1e9 #in ns
        bestMetricesJoint[dnn]['a']=bestMetricesJoint[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJoint[dnn]['e'],'mJ,  ', 'L =',bestMetricesJoint[dnn]['l'],'ns,  ', 'A =',bestMetricesJoint[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesJoint[dnn]['tp'], ',  TOPS/W =',bestMetricesJoint[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJoint[dnn]['a_tpm'])
    
    
    
   
    print('')
    print('THIS IS FOR MAX:')
    print('')
    
    #FOR MAX:
    #tests3=3
    dataMax={}
    keys_listMax={}
    dnn_names2=["vgg16"]
    dnn_names3=["resnet18","alexnet","mobilenet_v3"]
    for dnn in dnn_names2:
        path=folderSep+dnn+tosep+str(tests)+'.json'
        #path=folderSep+dnn+'/test'+str(tests3)+'.json'
        dataMax[dnn]=load_dict_from_json(path)
        keys_listMax[dnn] = list(dataMax[dnn].keys())
    
    
    print('')
    allSeparateScoresMAX={}
    percDiffMAX={}
    bestMetricesMax={}
    #for dnn in dnn_names:
    bestArchMax={}    
    for dnn in dnn_names2:
        reference_best=bestSeparateReferenceScore[dnn]
        scoresMax={}
        for keys in keys_listMax[dnn]:
            scoresMax[keys]=objective(dataMax[dnn][keys]['l'], dataMax[dnn][keys]['e'], dataMax[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar, obj_type)
        top_10SepMax = sorted(scoresMax.items(), key=lambda x: x[1], reverse=False)[:numBEST]
        print('dnn:', dnn)
        #print(top_10SepMax)
        print('key',top_10SepMax[0][0])
        print('value',top_10SepMax[0][1])
        #print('EDAP (mJ*ms*mm2):', bestMetricesMax[dnn]['e']*bestMetricesMax[dnn]['l']*bestMetricesMax[dnn]['a']*1e3*1e9*1e2*1e-6)
        best_Sep = top_10SepMax[0][0]  # Get the first key in the sorted dictionary
        bestArchMax[dnn]=best_Sep
        bestMetricesMax[dnn]=dataMax[dnn][best_Sep]
        score_bestSep = top_10SepMax[0][1]
        norm_score_bestSep=(score_bestSep- normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: norm_score_bestSep (normalized score to reference)', norm_score_bestSep)
        diffr=norm_score_bestSepForRef[dnn]-norm_score_bestSep
        diffrPers=diffr*100
        normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10SepMax]
        allSeparateScoresMAX[dnn]=normalized_joint
        percDiffMAX[dnn]=diffrPers

    first_elements = []
    topBestForAnalysis=[]
    for item in top_10SepMax:
        first_elements.append(item[0])
    for element in first_elements:
        parts = element.split('_')
        # Regular expression to match both integers and floats
        number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        # Extract numbers (integers or floats) from the split parts
        numbers = [part for part in parts if number_pattern.match(part)]
        #print(numbers)
        converted_values = [float(val) if 'e' in val or '.' in val else int(val) for val in numbers]
        topBestForAnalysis.append(converted_values)

    print('topBestForAnalysis',topBestForAnalysis)

    maxres=[]
    print('Analysing MAX')
    for keys in topBestForAnalysis:
        print('Evaluating', keys)
        hardware_params=keys
        dictts=runSingle(dnn_names3, num_iterations_to_run, num_mapping,n_jobs, hardware_params)
        maxres.append(dictts)
        
    #print('maxres',maxres)
    
    
    for dnn in dnn_names3:
        print('Analysing MAX', dnn)
        reference_best=bestSeparateReferenceScore[dnn]
        scoresMax=[]
        # Printing the first elements  
        for i, keys in enumerate(topBestForAnalysis):
            print('Evaluating', keys, 'i',  i)
            scoresMax.append(objective(maxres[i][dnn]['l'], maxres[i][dnn]['e'], maxres[i][dnn]['a'], coeff_lat, coeff_en, coeff_ar, obj_type))
            print('CHECK THIS FOR TOPS')
            print('a_tpw', maxres[i][dnn]['a_tpw'])
            print('a_tpm', maxres[i][dnn]['a_tpm'])
        print('scoresMax',scoresMax)
        print('dnn:', dnn)
        #print(top_10SepMax)
        print('key',topBestForAnalysis[0])
        print('value',scoresMax[0])
        print('EDAP (mJ*ms*mm2):', scoresMax[0]*1e3*1e9*1e2*1e-6)
        #top_10Sep = sorted(scoresMax.items(), key=lambda x: x[1], reverse=False)[:10]
        #print(top_10Sep)
        #print(top_10Sep[0])
        #print(top_10Sep[1])
        #best_Sep = top_10Sep[0][0]  # Get the first key in the sorted dictionary
        bestArchMax[dnn]=top_10SepMax[0][0]
        score_bestSep = scoresMax[0]
        norm_score_bestSep=(score_bestSep- normmin)/(reference_best - normmin)
        print('norm_score_bestSep (normalized score to reference)', norm_score_bestSep)
        diffr=norm_score_bestSep-best_scoresForEach[dnn]
        diffrPers=diffr*100
        normalized_joint = [(items - normmin) / (reference_best - normmin) for items in scoresMax]
        allSeparateScoresMAX[dnn]=normalized_joint
        percDiffMAX[dnn]=diffrPers
        
    
    #print('allSeparateScores')
    #print(allSeparateScores)
    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiffMAX)
    print('')
    print('best separate')
    for dnn in dnn_names2:
        print('   ', dnn)
        parts = bestArchMax[dnn].split('_')
        # Regular expression to match both integers and floats
        number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        # Extract numbers (integers or floats) from the split parts
        
        numbers = [part for part in parts if number_pattern.match(part)]
        #print(numbers)
        #converted_values = [float(val) if 'e' in val or '.' in val else int(val) for val in a]

        numbers[0]=numbers[0]+'V'
        numbers[1]=numbers[1]+'b'
        numbers[2]=numbers[2]+'s'
        numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
        print('Best params separate',numbers)
        
        bestMetricesMax[dnn]['e']=bestMetricesMax[dnn]['e']*1e3 #in mJ
        bestMetricesMax[dnn]['l']=bestMetricesMax[dnn]['l']*1e9 #in ns
        bestMetricesMax[dnn]['a']=bestMetricesMax[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesMax[dnn]['e'],'mJ,  ', 'L =',bestMetricesMax[dnn]['l'],'ns,  ', 'A =',bestMetricesMax[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesMax[dnn]['tp'], ',  TOPS/W =',bestMetricesMax[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesMax[dnn]['a_tpm'])
        print('EDAP (mJ*ms*mm2):', bestMetricesMax[dnn]['e']*bestMetricesMax[dnn]['l']*bestMetricesMax[dnn]['a']*1e-6)
        print('CHECK THIS FOR TOPS')
        print(bestMetricesMax[dnn])
    # PRINT STARTS HERE
    
    #print('log')
    #print([math.log(item) for item in allSeparateScores['resnet18']])

    
    print('')
    print('THIS IS FOR JOINT NEW ALG:')
    print('')
    

    dataNEW={}
    
    for dnn in dnn_names:
        pathNEW=folderNew+dnn+'/test'+str(tests)+ending
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
        
    top_10NEW = sorted(scoresNEW.items(), key=lambda x: x[1], reverse=False)[:numBEST]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_jointNEW=top_10NEW[0][0]
    reference_bestMaxedScoreNEW =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0


    percDiffNEW={}
    
    bestMetricesJointNEW={}
    bestJointReferenceScoreNEW={}
    for dnn in dnn_names:
        bestMetricesJointNEW[dnn]=dataNEW[dnn][best_jointNEW]
        bestJointReferenceScoreNEW[dnn]=objective(dataNEW[dnn][best_jointNEW]['l'], dataNEW[dnn][best_jointNEW]['e'], dataNEW[dnn][best_jointNEW]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keysNEW=[items[0] for items in top_10NEW]
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    best_scoresForEachNEW={}
    scores_separateJointNEW={}
    for dnn in dnn_names:
        reference_best=bestSeparateReferenceScore[dnn]
        scores2NEW={}
        for keys in top10_keysNEW:
            scores2NEW[keys]=objective(dataNEW[dnn][keys]['l'], dataNEW[dnn][keys]['e'], dataNEW[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        values_listNEW = [scores2NEW[key] for key in top10_keysNEW]
        #print(values_list)
        print('dnn:', dnn)
        print('best_Joint key:',top10_keysNEW[0],'best_Joint value:',values_listNEW[0])
        best_arch_scoreNEW=scores2NEW[best_jointNEW]
        normalized_valuesNEW = [(x - normmin) / (reference_best - normmin) for x in values_listNEW]
        scores_separateJointNEW[dnn]=normalized_valuesNEW
        best_scoresForEachNEW[dnn]=(best_arch_scoreNEW-normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEachNEW[dnn])
        diffrNEW=best_scoresForEachNEW[dnn]-norm_score_bestSepForRef[dnn]
        diffrPersNEW=diffrNEW*100
        normalized_jointNEW = [(items[1] - normmin) / (reference_best - normmin) for items in top_10NEW]
        #allSeparateScores[dnn]=normalized_joint
        percDiffNEW[dnn]=diffrPersNEW

    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiffNEW)
    print('')
    print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth')
    partsNEW = best_jointNEW.split('_')
    print(partsNEW)
    # Regular expression to match both integers and floats
    number_patternNEW = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbersNEW = [part for part in partsNEW if number_patternNEW.match(part)]
    numbersNEW[0]=numbersNEW[0]+'V'
    numbersNEW[1]=numbersNEW[1]+'b'
    numbersNEW[2]=numbersNEW[2]+'s'
    numbersNEW[8]=str(round(int(numbersNEW[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbersNEW)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJointNEW[dnn]['e']=bestMetricesJointNEW[dnn]['e']*1e3 #in mJ
        bestMetricesJointNEW[dnn]['l']=bestMetricesJointNEW[dnn]['l']*1e9 #in ns
        bestMetricesJointNEW[dnn]['a']=bestMetricesJointNEW[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJointNEW[dnn]['e'],'mJ,  ', 'L =',bestMetricesJointNEW[dnn]['l'],'ns,  ', 'A =',bestMetricesJointNEW[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesJointNEW[dnn]['tp'], ',  TOPS/W =',bestMetricesJointNEW[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJointNEW[dnn]['a_tpm'])
        print('EDAP (mJ*ms*mm2):', bestMetricesJointNEW[dnn]['e']*bestMetricesJointNEW[dnn]['l']*bestMetricesJointNEW[dnn]['a']*1e3*1e9*1e2*1e-6)
    
    print('')
    print('THIS IS FOR JOINT NEW ALG 2 :')
    print('')
    

    dataNEW2={}
    
    for dnn in dnn_names:
        pathNEW2=folderNew2+dnn+'/test'+str(tests)+ending
        dataNEW2[dnn]=load_dict_from_json(pathNEW2)
        if dnn==dnn_names[-1]:
            keys_listNEW2 = list(dataNEW2[dnn].keys())
    scoresNEW2={}
    for keys in keys_listNEW2:
        latencysNEW2=[dataNEW2[dnn][keys]['l'] for dnn in dnn_names]
        energysNEW2=[dataNEW2[dnn][keys]['e'] for dnn in dnn_names]
        areasNEW2=[dataNEW2[dnn][keys]['a'] for dnn in dnn_names]
        areaNEW2=max(areasNEW2)
        scoresNEW2[keys]=objectiveMAX(latencysNEW2, energysNEW2, areaNEW2, coeff_lat, coeff_en, coeff_ar,obj_type)
        
    top_10NEW2 = sorted(scoresNEW2.items(), key=lambda x: x[1], reverse=False)[:numBEST]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_jointNEW2=top_10NEW2[0][0]
    reference_bestMaxedScoreNEW2 =top_10NEW2[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0


    percDiffNEW2={}
    
    bestMetricesJointNEW2={}
    bestJointReferenceScoreNEW2={}
    for dnn in dnn_names:
        bestMetricesJointNEW2[dnn]=dataNEW2[dnn][best_jointNEW2]
        bestJointReferenceScoreNEW2[dnn]=objective(dataNEW2[dnn][best_jointNEW2]['l'], dataNEW2[dnn][best_jointNEW2]['e'], dataNEW2[dnn][best_jointNEW2]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keysNEW2=[items[0] for items in top_10NEW2]
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    best_scoresForEachNEW2={}
    scores_separateJointNEW2={}
    for dnn in dnn_names:
        reference_best=bestSeparateReferenceScore[dnn]
        scores2NEW2={}
        for keys in top10_keysNEW2:
            scores2NEW2[keys]=objective(dataNEW2[dnn][keys]['l'], dataNEW2[dnn][keys]['e'], dataNEW2[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        values_listNEW2 = [scores2NEW2[key] for key in top10_keysNEW2]
        #print(values_list)
        print('dnn:', dnn)
        print('best_Joint key:',top10_keysNEW2[0],'best_Joint value:',values_listNEW2[0])
        best_arch_scoreNEW2=scores2NEW2[best_jointNEW2]
        normalized_valuesNEW2 = [(x - normmin) / (reference_best - normmin) for x in values_listNEW2]
        scores_separateJointNEW2[dnn]=normalized_valuesNEW2
        best_scoresForEachNEW2[dnn]=(best_arch_scoreNEW2-normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEachNEW2[dnn])
        diffrNEW2=best_scoresForEachNEW2[dnn]-norm_score_bestSepForRef[dnn]
        diffrPersNEW2=diffrNEW2*100
        normalized_jointNEW2 = [(items[1] - normmin) / (reference_best - normmin) for items in top_10NEW2]
        #allSeparateScores[dnn]=normalized_joint
        percDiffNEW2[dnn]=diffrPersNEW2

    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiffNEW2)
    print('')
    print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth')
    partsNEW2 = best_jointNEW2.split('_')
    print(partsNEW2)
    # Regular expression to match both integers and floats
    number_patternNEW2 = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbersNEW2 = [part for part in partsNEW2 if number_patternNEW2.match(part)]
    numbersNEW2[0]=numbersNEW2[0]+'V'
    numbersNEW2[1]=numbersNEW2[1]+'b'
    numbersNEW2[2]=numbersNEW2[2]+'s'
    numbersNEW2[8]=str(round(int(numbersNEW2[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbersNEW2)
    #print('EDAP (mJ*ms*mm2):', bestMetricesMax[dnn]['e']*bestMetricesMax[dnn]['l']*bestMetricesMax[dnn]['a']*1e3*1e9*1e2*1e-6)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJointNEW2[dnn]['e']=bestMetricesJointNEW2[dnn]['e']*1e3 #in mJ
        bestMetricesJointNEW2[dnn]['l']=bestMetricesJointNEW2[dnn]['l']*1e9 #in ns
        bestMetricesJointNEW2[dnn]['a']=bestMetricesJointNEW2[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJointNEW2[dnn]['e'],'mJ,  ', 'L =',bestMetricesJointNEW2[dnn]['l'],'ns,  ', 'A =',bestMetricesJointNEW2[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesJointNEW2[dnn]['tp'], ',  TOPS/W =',bestMetricesJointNEW2[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJointNEW2[dnn]['a_tpm'])
        print('EDAP (mJ*ms*mm2):', bestMetricesJointNEW2[dnn]['e']*bestMetricesJointNEW2[dnn]['l']*bestMetricesJointNEW2[dnn]['a']*1e-6)
        print('CHECK THIS FOR TOPS')
        print(bestMetricesSep[dnn])
    
    
    
    # HERE PLOTTING STARTS!!!!!
    
    # Define separate lists for x and y values where x has fewer unique elements compared to y
    #x_values = [1, 2, 3,4,5,7,8,9,10,11, 13, 14,15, 16,17,19,20,21,22,23]  # Unique x values
    x_values = [1, 2, 3,4,5,8,9,10,11,12, 15, 16,17,18,19,22,23,24,25,26]  # Unique x values
    
    percentage=[str(abs(round(percDiff[dnn])))+'%' for dnn in dnn_names]
    y_values = [
        
        allSeparateScores[dnn_names[0]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[0]],
        scores_separateJoint[dnn_names[0]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[0]], 
        scores_separateJointNEW2[dnn_names[0]], 
        
        
        allSeparateScores[dnn_names[1]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[1]],
        scores_separateJoint[dnn_names[1]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[1]], 
        scores_separateJointNEW2[dnn_names[1]], 
        
        allSeparateScores[dnn_names[2]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[2]],
        scores_separateJoint[dnn_names[2]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[2]], 
        scores_separateJointNEW2[dnn_names[2]], 
        
        allSeparateScores[dnn_names[3]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[3]],
        scores_separateJoint[dnn_names[3]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[3]],
        scores_separateJointNEW2[dnn_names[3]]
    ]
    
    #x_labels = ['Joint'+dnn_names[0], 'Separate'+dnn_names[0],'Joint'+dnn_names[1], 'Separate'+dnn_names[1],'Joint'+dnn_names[2], 'Separate'+dnn_names[2],'Joint'+dnn_names[3], 'Separate'+dnn_names[3]]
    x_labels = ['Separate', 'Max','Joint','JointNew2','JointNew4', 'Separate', 'Max','Joint','JointNew2','JointNew4', 'Separate', 'Max','Joint','JointNew2','JointNew4' ,'Separate', 'Max','Joint','JointNew2','JointNew4']
    
    # Define custom hex colors for each group
    #hex_colors = ['#76d3d7', '#76d3d7', '#a8e436', '#a8e436', '#a3cbef','#a3cbef', '#ffe822', '#ffe822']  # Blue, Green, and Orange shades
    #edge_colors=['#55979a', '#55979a', '#658922', '#658922', '#647e95','#647e95', '#9e901a', '#9e901a']
    hex_colors = ['#4361ee','#4361ee','#4361ee','#4361ee','#4361ee', '#20d4f6', '#20d4f6','#20d4f6','#20d4f6','#20d4f6','#d242ff','#d242ff','#d242ff', '#d242ff', '#d242ff', '#2a9d8f', '#2a9d8f','#2a9d8f', '#2a9d8f', '#2a9d8f']  # Blue, Green, and Orange shades
    edge_colors=['#32267d','#32267d', '#32267d','#32267d', '#32267d','#2e7294','#2e7294','#2e7294', '#2e7294','#2e7294','#470080','#470080','#470080','#470080','#470080', '#264653', '#264653','#264653', '#264653', '#264653']

    hex_colors = ['#4361ee','#20d4f6','#d242ff', '#2a9d8f', '#FF5400','#4361ee','#20d4f6','#d242ff', '#2a9d8f', '#FF5400','#4361ee','#20d4f6','#d242ff', '#2a9d8f','#FF5400','#4361ee','#20d4f6','#d242ff', '#2a9d8f', '#FF5400']  # Blue, Green, and Orange shades
    edge_colors=['#32267d','#2e7294','#470080','#264653', '#2E2B26','#32267d','#2e7294','#470080','#264653', '#2E2B26','#32267d','#2e7294','#470080','#264653', '#2E2B26','#32267d','#2e7294','#470080','#264653', '#2E2B26']

    #hex_colors = ["#BBA7FF","#098E87", "#04FBD2", "#08A8FF","#39DE2A","#BBA7FF","#098E87", "#04FBD2", "#08A8FF","#39DE2A","#BBA7FF","#098E87", "#04FBD2", "#08A8FF","#39DE2A","#BBA7FF","#098E87", "#04FBD2", "#08A8FF","#39DE2A"]
    #edge_colors=['black','black','black','black','black','black','black','black','black','black','black', 'black','black','black','black','black','black','black','black','black']

    
    #hex_colors = ["#BBA7FF","#FFE208", "#04FBD2", "#08A8FF","#39DE2A","#BBA7FF","#FFE208", "#04FBD2", "#08A8FF","#39DE2A","#BBA7FF","#FFE208", "#04FBD2", "#08A8FF","#39DE2A","#BBA7FF","#FFE208", "#04FBD2", "#08A8FF","#39DE2A"]

    #hex_colors = ["#FFE208","#39DE2A", "#BBA7FF","#04FBD2","#F38005", "#FFE208","#39DE2A", "#BBA7FF","#04FBD2","#F38005","#FFE208","#39DE2A", "#BBA7FF","#04FBD2","#F38005","#FFE208","#39DE2A", "#BBA7FF","#04FBD2","#F38005"]

    #hex_colors = ["#009F9F", "#A6FF00","#FF5400", "#0057FF", "#A000A0","#009F9F", "#A6FF00","#FF5400", "#FF0000", "#A000A0","#009F9F", "#A6FF00","#FF5400", "#FF0000", "#A000A0","#009F9F", "#A6FF00","#FF5400", "#FF0000", "#A000A0"]

    
    # Flatten the x and y lists to create a single list for plotting
    x_flat = [x for x, y_list in zip(x_values, y_values) for _ in y_list]  # Repeat each x according to the y elements
    y_flat = [y for y_list in y_values for y in y_list]  # Flatten all y values
    
    # Create the scatter plot
    
    fig1=plt.figure(figsize=(8, 6))
    
    # Plot each group with the specified hex colors and highlight the score with a star
    #for i, (x, y_vals) in enumerate(zip(x_values, y_values)):
        #if (i + 1) % 5 == 1:
            #plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='^',s=300, label=f'Group {x_labels[i]}')
        #elif (i + 1) % 5 == 2:
            #plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='s',s=300, label=f'Group {x_labels[i]}')
        #elif (i + 1) % 5 == 3:
            #plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='o',s=300, label=f'Group {x_labels[i]}')
        #elif (i + 1) % 5 == 4:
        #    plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='<',s=300, label=f'Group {x_labels[i]}')
       # else:
        #    plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='v',s=300, label=f'Group {x_labels[i]}')
        #plt.scatter(x, y_vals[0], color=hex_colors[i],edgecolors='black', marker='*', s=600)  # 's' controls the size of the star
    

    for i, (x, y_vals) in enumerate(zip(x_values, y_values)):
        if (i + 1) % 5 == 1:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='o',alpha=1,s=300, label=f'Group {x_labels[i]}')
        elif (i + 1) % 5 == 2:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='o',alpha=1,s=300, label=f'Group {x_labels[i]}')
        elif (i + 1) % 5 == 3:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='o',alpha=1,s=300, label=f'Group {x_labels[i]}')
        elif (i + 1) % 5 == 4:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='o',alpha=1,s=300, label=f'Group {x_labels[i]}')
        else:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='o',alpha=1,s=300, label=f'Group {x_labels[i]}')
        plt.scatter(x, y_vals[0], color=hex_colors[i],edgecolors='black', marker='*',alpha=1, s=600)  # 's' controls the size of the star

        
        # Highlight one dot per x value as a star with the same hex color  '#0E3C34'
        
        # Add a custom legend text directly on the graph next to the first star point of each group
        
        # Add a custom legend text directly on the graph for even-numbered groups only
        #if (i + 1) % 2 == 0:  # Check if the group is even (i+1 because index i starts from 0)
        #    plt.text(x + 0.3, y_vals[0], f'{percentage[round(np.floor(i/2))]}', fontsize=10, color=hex_colors[i], va='center')
    
        #highlighted_index = (i + 1) % 5  # Determines the shape for all points in the batch
        
        #if highlighted_index == 1:
         #   marker_type = '^'  # Star marker
        #elif highlighted_index == 2:
        #    marker_type = 's'  # Square
        #elif highlighted_index == 3:
        #    marker_type = 'o'  # Circle
        #elif highlighted_index == 4:
        #    marker_type = '<'  # Left triangle
        #else:
        #    marker_type = 'v'  # Down triangle
    
        # Highlight only the best value with a red border
        #plt.scatter(
        #    x, y_vals[0],  # The best value
        #    color=hex_colors[i], 
        #    edgecolors='black',  # Red border for highlighting
        #    marker=marker_type,  # Keep the same shape as others
        #    s=400,  # Make it larger for visibility
        ##    linewidths=2
        #)
  
    # Adding custom legends
    custom_legends = [
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[0], markersize=10, label=dnn_names[0]),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[2], markersize=10, label=dnn_names[1]),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[4], markersize=10, label=dnn_names[2]),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[6], markersize=10, label=dnn_names[3]),
        #plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#0E3C34', markersize=15, label='Best architecture')
    ]
    
    #plt.xlim(0, 24)  # Start from 0 and end at 4 on x-axis
    plt.xlim(0, 27)
    plt.ylim(0, ylimit) 
    #plt.xticks([])
    #plt.yticks([])
    #plt.legend(handles=custom_legends)
    # Adding labels and title
    #plt.xlabel('X-axis')
    #plt.ylabel('Normalized score')
    #plt.title('Scatter Plot with Separate X and Y Lists')
    plt.xticks(ticks=x_values, labels=x_labels)
    #plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.grid(axis='x')
    #plt.grid(axis='y')
    # Display the plot
    plt.tight_layout()
    plt.show()
    fig1.savefig(figname,dpi=2000)


def display_results6(folder, folderSep,folderNew, folderNew2, folderNew3, obj_type, constr_type, tests, figname, ylimit, n_jobs, numBEST, num_iterations_to_run=1, num_mapping=None):

    #folder='res/new/results_part1/unconstraint/joint/joint_L_noconstr/'
    #folderSep='res/new/results_part1/unconstraint/separate/separate_L_noconstr/'
    #folder='res/resnew/joint_EL_noconstr/'
    #obj_type='l'
    #constr_type='c'
    #tests=0 #this is for a single case!
    
    if constr_type=='n':
        ending='.json'
        tosep='/test'
    elif constr_type=='c':
        ending='Constr.json'
        tosep='/Constrtest'
    
    #folder='res/resnew/joint_A_nonconstr/'
    #folderSep='res/to_plot/first/allseparateRes/'
    dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3"]
    #testnum=8
    
    #pltcases={}
    #for tests in range(testnum):
    pltcase={}
    #print('')
    #print('')
    #print('Test',tests)
    #print('A: voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, shared_router_groupSize, tiles_in_chip, macros_in_tile, glb_buffer_depth')
    data={}
    
    for dnn in dnn_names:
        path=folder+dnn+'/test'+str(tests)+ending
        data[dnn]=load_dict_from_json(path)
        if dnn==dnn_names[-1]:
            keys_list = list(data[dnn].keys())
    scores={}
    for keys in keys_list:
        latencys=[data[dnn][keys]['l'] for dnn in dnn_names]
        energys=[data[dnn][keys]['e'] for dnn in dnn_names]
        areas=[data[dnn][keys]['a'] for dnn in dnn_names]
        area=max(areas)
        scores[keys]=objectiveMAX(latencys, energys, area, coeff_lat, coeff_en, coeff_ar,obj_type)
        
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=False)[:numBEST]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_joint=top_10[0][0]
    reference_bestMaxedScore =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0
    
    
    # HERE IS SEPARATE:
    print('')
    print('THIS IS FOR SEPARATE:')
    print('')

     #tests3=3
    dataSep={}
    keys_listSep={}
    
    for dnn in dnn_names:
        path=folderSep+dnn+tosep+str(tests)+'.json'
        #path=folderSep+dnn+'/test'+str(tests3)+'.json'
        dataSep[dnn]=load_dict_from_json(path)
        keys_listSep[dnn] = list(dataSep[dnn].keys())
    
    allSeparateScores={}
    percDiff={}
    bestMetricesSep={}
    #for dnn in dnn_names:
    bestArchSep={}
    bestSeparateReferenceScore={}
    norm_score_bestSepForRef={}
    for dnn in dnn_names:
        #reference_best=bestJointReferenceScore[dnn]
        scoresSep={}
        for keys in keys_listSep[dnn]:
            scoresSep[keys]=objective(dataSep[dnn][keys]['l'], dataSep[dnn][keys]['e'], dataSep[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar, obj_type)
        top_10Sep = sorted(scoresSep.items(), key=lambda x: x[1], reverse=False)[:numBEST]
        print('dnn:', dnn)
        #print('best_Sep key',top_10Sep[0][0])
        print('best_Sep value',top_10Sep[0][1])
        best_Sep = top_10Sep[0][0]  # Get the first key in the sorted dictionary
        bestSeparateReferenceScore[dnn]=top_10Sep[0][1]
        reference_best=bestSeparateReferenceScore[dnn]
        
        bestArchSep[dnn]=best_Sep
        bestMetricesSep[dnn]=dataSep[dnn][best_Sep]
        score_bestSep = top_10Sep[0][1]
        norm_score_bestSep=(score_bestSep- normmin)/(reference_best - normmin)
        norm_score_bestSepForRef[dnn]=norm_score_bestSep
        diffr=norm_score_bestSep-norm_score_bestSep
        diffrPers=diffr*100
        normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10Sep]
        allSeparateScores[dnn]=normalized_joint
        print('allSeparateScores[dnn]:', allSeparateScores[dnn])
        percDiff[dnn]=diffrPers
    #print('allSeparateScores')
    #print(allSeparateScores)
    print('')
    print('Percentage difference in score (negative = joint is better): separate')
    print(percDiff)
    print('')
    print('best separate')
    for dnn in dnn_names:
        print('   ', dnn)
        parts = bestArchSep[dnn].split('_')
        # Regular expression to match both integers and floats
        number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        # Extract numbers (integers or floats) from the split parts
        numbers = [part for part in parts if number_pattern.match(part)]
        numbers[0]=numbers[0]+'V'
        numbers[1]=numbers[1]+'b'
        numbers[2]=numbers[2]+'s'
        numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
        print('Best params separate',numbers)
        
        bestMetricesSep[dnn]['e']=bestMetricesSep[dnn]['e']*1e3 #in mJ
        bestMetricesSep[dnn]['l']=bestMetricesSep[dnn]['l']*1e9 #in ns
        bestMetricesSep[dnn]['a']=bestMetricesSep[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesSep[dnn]['e'],'mJ,  ', 'L =',bestMetricesSep[dnn]['l'],'ns,  ', 'A =',bestMetricesSep[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesSep[dnn]['tp'], ',  TOPS/W =',bestMetricesSep[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesSep[dnn]['a_tpm'])



    
    print('')
    print('THIS IS FOR JOINT:')
    print('')
    #HERE IS NORMALIZATION FOR JOINT:
    
    bestMetricesJoint={}
    bestJointReferenceScore={}
    for dnn in dnn_names:
        bestMetricesJoint[dnn]=data[dnn][best_joint]
        bestJointReferenceScore[dnn]=objective(data[dnn][best_joint]['l'], data[dnn][best_joint]['e'], data[dnn][best_joint]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keys=[items[0] for items in top_10]
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    best_scoresForEach={}
    scores_separateJoint={}
    for dnn in dnn_names:
        reference_best=bestSeparateReferenceScore[dnn]
        scores2={}
        for keys in top10_keys:
            scores2[keys]=objective(data[dnn][keys]['l'], data[dnn][keys]['e'], data[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        values_list = [scores2[key] for key in top10_keys]
        #print(values_list)
        print('dnn:', dnn)
        print('best_Joint key:',top10_keys[0],'best_Joint value:',values_list[0])
        best_arch_score=scores2[best_joint]
        normalized_values = [(x - normmin) / (reference_best - normmin) for x in values_list]
        scores_separateJoint[dnn]=normalized_values
        best_scoresForEach[dnn]=(best_arch_score-normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEach[dnn])
        diffr=best_scoresForEach[dnn]-norm_score_bestSepForRef[dnn]
        diffrPers=diffr*100
        normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
        #allSeparateScores[dnn]=normalized_joint
        percDiff[dnn]=diffrPers

    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiff)
    print('')
    print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth')
    parts = best_joint.split('_')
    print('parts', parts)
    # Regular expression to match both integers and floats
    number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbers = [part for part in parts if number_pattern.match(part)]
    numbers[0]=numbers[0]+'V'
    numbers[1]=numbers[1]+'b'
    numbers[2]=numbers[2]+'s'
    numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbers)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJoint[dnn]['e']=bestMetricesJoint[dnn]['e']*1e3 #in mJ
        bestMetricesJoint[dnn]['l']=bestMetricesJoint[dnn]['l']*1e9 #in ns
        bestMetricesJoint[dnn]['a']=bestMetricesJoint[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJoint[dnn]['e'],'mJ,  ', 'L =',bestMetricesJoint[dnn]['l'],'ns,  ', 'A =',bestMetricesJoint[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesJoint[dnn]['tp'], ',  TOPS/W =',bestMetricesJoint[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJoint[dnn]['a_tpm'])
    
    
    
   
    print('')
    print('THIS IS FOR MAX:')
    print('')
    
    #FOR MAX:
    #tests3=3
    dataMax={}
    keys_listMax={}
    dnn_names2=["vgg16"]
    dnn_names3=["resnet18","alexnet","mobilenet_v3"]
    for dnn in dnn_names2:
        path=folderSep+dnn+tosep+str(tests)+'.json'
        #path=folderSep+dnn+'/test'+str(tests3)+'.json'
        dataMax[dnn]=load_dict_from_json(path)
        keys_listMax[dnn] = list(dataMax[dnn].keys())
    
    
    print('')
    allSeparateScoresMAX={}
    percDiffMAX={}
    bestMetricesMax={}
    #for dnn in dnn_names:
    bestArchMax={}    
    for dnn in dnn_names2:
        reference_best=bestSeparateReferenceScore[dnn]
        scoresMax={}
        for keys in keys_listMax[dnn]:
            scoresMax[keys]=objective(dataMax[dnn][keys]['l'], dataMax[dnn][keys]['e'], dataMax[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar, obj_type)
        top_10SepMax = sorted(scoresMax.items(), key=lambda x: x[1], reverse=False)[:numBEST]
        print('dnn:', dnn)
        #print(top_10SepMax)
        print('key',top_10SepMax[0][0])
        print('value',top_10SepMax[0][1])
        print('EDAP (mJ*ms*mm2):', top_10SepMax[0][1]*1e3*1e9*1e2*1e-6)
        best_Sep = top_10SepMax[0][0]  # Get the first key in the sorted dictionary
        bestArchMax[dnn]=best_Sep
        bestMetricesMax[dnn]=dataMax[dnn][best_Sep]
        score_bestSep = top_10SepMax[0][1]
        norm_score_bestSep=(score_bestSep- normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: norm_score_bestSep (normalized score to reference)', norm_score_bestSep)
        diffr=norm_score_bestSepForRef[dnn]-norm_score_bestSep
        diffrPers=diffr*100
        normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10SepMax]
        allSeparateScoresMAX[dnn]=normalized_joint
        percDiffMAX[dnn]=diffrPers

    first_elements = []
    topBestForAnalysis=[]
    for item in top_10SepMax:
        first_elements.append(item[0])
    for element in first_elements:
        parts = element.split('_')
        # Regular expression to match both integers and floats
        number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        # Extract numbers (integers or floats) from the split parts
        numbers = [part for part in parts if number_pattern.match(part)]
        #print(numbers)
        converted_values = [float(val) if 'e' in val or '.' in val else int(val) for val in numbers]
        topBestForAnalysis.append(converted_values)

    print('topBestForAnalysis',topBestForAnalysis)

    maxres=[]
    print('Analysing MAX')
    for keys in topBestForAnalysis:
        print('Evaluating', keys)
        hardware_params=keys
        dictts=runSingle(dnn_names3, num_iterations_to_run, num_mapping,n_jobs, hardware_params)
        maxres.append(dictts)
        
    #print('maxres',maxres)
    
    
    for dnn in dnn_names3:
        print('Analysing MAX', dnn)
        reference_best=bestSeparateReferenceScore[dnn]
        scoresMax=[]
        # Printing the first elements  
        for i, keys in enumerate(topBestForAnalysis):
            print('Evaluating', keys, 'i',  i)
            scoresMax.append(objective(maxres[i][dnn]['l'], maxres[i][dnn]['e'], maxres[i][dnn]['a'], coeff_lat, coeff_en, coeff_ar, obj_type))
        
        print('scoresMax',scoresMax)
        print('dnn:', dnn)
        #print(top_10SepMax)
        print('key',topBestForAnalysis[0])
        print('value',scoresMax[0])
        print('EDAP (mJ*ms*mm2):', scoresMax[0]*1e3*1e9*1e2*1e-6)

        #top_10Sep = sorted(scoresMax.items(), key=lambda x: x[1], reverse=False)[:10]
        #print(top_10Sep)
        #print(top_10Sep[0])
        #print(top_10Sep[1])
        #best_Sep = top_10Sep[0][0]  # Get the first key in the sorted dictionary
        bestArchMax[dnn]=top_10SepMax[0][0]
        score_bestSep = scoresMax[0]
        norm_score_bestSep=(score_bestSep- normmin)/(reference_best - normmin)
        print('norm_score_bestSep (normalized score to reference)', norm_score_bestSep)
        diffr=norm_score_bestSep-best_scoresForEach[dnn]
        diffrPers=diffr*100
        normalized_joint = [(items - normmin) / (reference_best - normmin) for items in scoresMax]
        allSeparateScoresMAX[dnn]=normalized_joint
        percDiffMAX[dnn]=diffrPers
    
    #print('allSeparateScores')
    #print(allSeparateScores)
    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiffMAX)
    print('')
    print('best separate')
    for dnn in dnn_names2:
        print('   ', dnn)
        parts = bestArchMax[dnn].split('_')
        # Regular expression to match both integers and floats
        number_pattern = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
        # Extract numbers (integers or floats) from the split parts
        
        numbers = [part for part in parts if number_pattern.match(part)]
        #print(numbers)
        #converted_values = [float(val) if 'e' in val or '.' in val else int(val) for val in a]

        numbers[0]=numbers[0]+'V'
        numbers[1]=numbers[1]+'b'
        numbers[2]=numbers[2]+'s'
        numbers[8]=str(round(int(numbers[8])*2048/1024/1024/8))+'MB'
        print('Best params separate',numbers)
        
        bestMetricesMax[dnn]['e']=bestMetricesMax[dnn]['e']*1e3 #in mJ
        bestMetricesMax[dnn]['l']=bestMetricesMax[dnn]['l']*1e9 #in ns
        bestMetricesMax[dnn]['a']=bestMetricesMax[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesMax[dnn]['e'],'mJ,  ', 'L =',bestMetricesMax[dnn]['l'],'ns,  ', 'A =',bestMetricesMax[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesMax[dnn]['tp'], ',  TOPS/W =',bestMetricesMax[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesMax[dnn]['a_tpm'])
        print('EDAP (mJ*ms*mm2):', bestMetricesMax[dnn]['e']*bestMetricesMax[dnn]['l']*bestMetricesMax[dnn]['a']*1e-6)

    # PRINT STARTS HERE
    
    #print('log')
    #print([math.log(item) for item in allSeparateScores['resnet18']])

    
    print('')
    print('THIS IS FOR JOINT NEW ALG:')
    print('')
    

    dataNEW={}
    
    for dnn in dnn_names:
        pathNEW=folderNew+dnn+'/test'+str(tests)+ending
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
        
    top_10NEW = sorted(scoresNEW.items(), key=lambda x: x[1], reverse=False)[:numBEST]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_jointNEW=top_10NEW[0][0]
    reference_bestMaxedScoreNEW =top_10[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0


    percDiffNEW={}
    
    bestMetricesJointNEW={}
    bestJointReferenceScoreNEW={}
    for dnn in dnn_names:
        bestMetricesJointNEW[dnn]=dataNEW[dnn][best_jointNEW]
        bestJointReferenceScoreNEW[dnn]=objective(dataNEW[dnn][best_jointNEW]['l'], dataNEW[dnn][best_jointNEW]['e'], dataNEW[dnn][best_jointNEW]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keysNEW=[items[0] for items in top_10NEW]
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    best_scoresForEachNEW={}
    scores_separateJointNEW={}
    for dnn in dnn_names:
        reference_best=bestSeparateReferenceScore[dnn]
        scores2NEW={}
        for keys in top10_keysNEW:
            scores2NEW[keys]=objective(dataNEW[dnn][keys]['l'], dataNEW[dnn][keys]['e'], dataNEW[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        values_listNEW = [scores2NEW[key] for key in top10_keysNEW]
        #print(values_list)
        print('dnn:', dnn)
        print('best_Joint key:',top10_keysNEW[0],'best_Joint value:',values_listNEW[0])
        best_arch_scoreNEW=scores2NEW[best_jointNEW]
        normalized_valuesNEW = [(x - normmin) / (reference_best - normmin) for x in values_listNEW]
        scores_separateJointNEW[dnn]=normalized_valuesNEW
        best_scoresForEachNEW[dnn]=(best_arch_scoreNEW-normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEachNEW[dnn])
        diffrNEW=best_scoresForEachNEW[dnn]-norm_score_bestSepForRef[dnn]
        diffrPersNEW=diffrNEW*100
        normalized_jointNEW = [(items[1] - normmin) / (reference_best - normmin) for items in top_10NEW]
        #allSeparateScores[dnn]=normalized_joint
        percDiffNEW[dnn]=diffrPersNEW

    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiffNEW)
    print('')
    print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth')
    partsNEW = best_jointNEW.split('_')
    print(partsNEW)
    # Regular expression to match both integers and floats
    number_patternNEW = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbersNEW = [part for part in partsNEW if number_patternNEW.match(part)]
    numbersNEW[0]=numbersNEW[0]+'V'
    numbersNEW[1]=numbersNEW[1]+'b'
    numbersNEW[2]=numbersNEW[2]+'s'
    numbersNEW[8]=str(round(int(numbersNEW[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbersNEW)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJointNEW[dnn]['e']=bestMetricesJointNEW[dnn]['e']*1e3 #in mJ
        bestMetricesJointNEW[dnn]['l']=bestMetricesJointNEW[dnn]['l']*1e9 #in ns
        bestMetricesJointNEW[dnn]['a']=bestMetricesJointNEW[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJointNEW[dnn]['e'],'mJ,  ', 'L =',bestMetricesJointNEW[dnn]['l'],'ns,  ', 'A =',bestMetricesJointNEW[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesJointNEW[dnn]['tp'], ',  TOPS/W =',bestMetricesJointNEW[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJointNEW[dnn]['a_tpm'])
        print('EDAP (mJ*ms*mm2):', bestMetricesMax[dnn]['e']*bestMetricesMax[dnn]['l']*bestMetricesMax[dnn]['a']*1e3*1e9*1e2*1e-6)
    
    print('')
    print('THIS IS FOR JOINT NEW ALG 2 :')
    print('')
    

    dataNEW2={}
    
    for dnn in dnn_names:
        pathNEW2=folderNew2+dnn+'/test'+str(tests)+ending
        dataNEW2[dnn]=load_dict_from_json(pathNEW2)
        if dnn==dnn_names[-1]:
            keys_listNEW2 = list(dataNEW2[dnn].keys())
    scoresNEW2={}
    for keys in keys_listNEW2:
        latencysNEW2=[dataNEW2[dnn][keys]['l'] for dnn in dnn_names]
        energysNEW2=[dataNEW2[dnn][keys]['e'] for dnn in dnn_names]
        areasNEW2=[dataNEW2[dnn][keys]['a'] for dnn in dnn_names]
        areaNEW2=max(areasNEW2)
        scoresNEW2[keys]=objectiveMAX(latencysNEW2, energysNEW2, areaNEW2, coeff_lat, coeff_en, coeff_ar,obj_type)
        
    top_10NEW2 = sorted(scoresNEW2.items(), key=lambda x: x[1], reverse=False)[:numBEST]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_jointNEW2=top_10NEW2[0][0]
    reference_bestMaxedScoreNEW2 =top_10NEW2[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0


    percDiffNEW2={}
    
    bestMetricesJointNEW2={}
    bestJointReferenceScoreNEW2={}
    for dnn in dnn_names:
        bestMetricesJointNEW2[dnn]=dataNEW2[dnn][best_jointNEW2]
        bestJointReferenceScoreNEW2[dnn]=objective(dataNEW2[dnn][best_jointNEW2]['l'], dataNEW2[dnn][best_jointNEW2]['e'], dataNEW2[dnn][best_jointNEW2]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keysNEW2=[items[0] for items in top_10NEW2]
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    best_scoresForEachNEW2={}
    scores_separateJointNEW2={}
    for dnn in dnn_names:
        reference_best=bestSeparateReferenceScore[dnn]
        scores2NEW2={}
        for keys in top10_keysNEW2:
            scores2NEW2[keys]=objective(dataNEW2[dnn][keys]['l'], dataNEW2[dnn][keys]['e'], dataNEW2[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        values_listNEW2 = [scores2NEW2[key] for key in top10_keysNEW2]
        #print(values_list)
        print('dnn:', dnn)
        print('best_Joint key:',top10_keysNEW2[0],'best_Joint value:',values_listNEW2[0])
        best_arch_scoreNEW2=scores2NEW2[best_jointNEW2]
        normalized_valuesNEW2 = [(x - normmin) / (reference_best - normmin) for x in values_listNEW2]
        scores_separateJointNEW2[dnn]=normalized_valuesNEW2
        best_scoresForEachNEW2[dnn]=(best_arch_scoreNEW2-normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEachNEW2[dnn])
        diffrNEW2=best_scoresForEachNEW2[dnn]-norm_score_bestSepForRef[dnn]
        diffrPersNEW2=diffrNEW2*100
        normalized_jointNEW2 = [(items[1] - normmin) / (reference_best - normmin) for items in top_10NEW2]
        #allSeparateScores[dnn]=normalized_joint
        percDiffNEW2[dnn]=diffrPersNEW2

    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiffNEW2)
    print('')
    print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth')
    partsNEW2 = best_jointNEW2.split('_')
    print(partsNEW2)
    # Regular expression to match both integers and floats
    number_patternNEW2 = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbersNEW2 = [part for part in partsNEW2 if number_patternNEW2.match(part)]
    numbersNEW2[0]=numbersNEW2[0]+'V'
    numbersNEW2[1]=numbersNEW2[1]+'b'
    numbersNEW2[2]=numbersNEW2[2]+'s'
    numbersNEW2[8]=str(round(int(numbersNEW2[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbersNEW2)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJointNEW2[dnn]['e']=bestMetricesJointNEW2[dnn]['e']*1e3 #in mJ
        bestMetricesJointNEW2[dnn]['l']=bestMetricesJointNEW2[dnn]['l']*1e9 #in ns
        bestMetricesJointNEW2[dnn]['a']=bestMetricesJointNEW2[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJointNEW2[dnn]['e'],'mJ,  ', 'L =',bestMetricesJointNEW2[dnn]['l'],'ns,  ', 'A =',bestMetricesJointNEW2[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesJointNEW2[dnn]['tp'], ',  TOPS/W =',bestMetricesJointNEW2[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJointNEW2[dnn]['a_tpm'])
    
    print('')
    print('THIS IS FOR JOINT NEW ALG 2 :')
    print('')
    

    dataNEW3={}
    
    for dnn in dnn_names:
        pathNEW3=folderNew3+dnn+'/test'+str(tests)+ending
        dataNEW3[dnn]=load_dict_from_json(pathNEW3)
        if dnn==dnn_names[-1]:
            keys_listNEW3 = list(dataNEW3[dnn].keys())
    scoresNEW3={}
    for keys in keys_listNEW3:
        latencysNEW3=[dataNEW3[dnn][keys]['l'] for dnn in dnn_names]
        energysNEW3=[dataNEW3[dnn][keys]['e'] for dnn in dnn_names]
        areasNEW3=[dataNEW3[dnn][keys]['a'] for dnn in dnn_names]
        areaNEW3=max(areasNEW3)
        scoresNEW3[keys]=objectiveMAX(latencysNEW3, energysNEW3, areaNEW3, coeff_lat, coeff_en, coeff_ar,obj_type)
        
    top_10NEW3 = sorted(scoresNEW3.items(), key=lambda x: x[1], reverse=False)[:numBEST]
    
    #defining minimum possible score for the best selected architectures in a joint search:
    #print('top_10 joint',top_10)
    # Display the top 10
    #best_jointX = next(iter(top_10))  # Get the first key in the sorted dictionary
    best_jointNEW3=top_10NEW3[0][0]
    reference_bestMaxedScoreNEW3 =top_10NEW3[0][1]
    #print('reference_bestMaxedScore')
    #print(reference_bestMaxedScore)
    normmin=0


    percDiffNEW3={}
    #from plotallWithMaxNewAlg import *
    #tutorials/resHWC_NEW/SRAM/separate_ELA_constr8_seed1/gpt2_medium/Constrtest0.json
    #dnn_names=["resnet18","vgg16","alexnet","mobilenet_v3","mobilebert","densenet201","resnet50","vision_transformer","gpt2_medium"]
    #tutorials/resHWC/SRAM_DRAM/joint_ELA_constr8_seed1
    #folder='resHWC_NEW/SRAM_hammigDist4Phase/joint_ELA_constr8_seed1_forDropGraph40/'
    #obj_type='ela_mean'
    #constr_type='c'
    #tests=0 #this is for a single case!
    #display_resultsSINGLEforSEED_specific(folder, obj_type, constr_type, tests,dnn_names)
    #display_resultsCOST(folder, obj_type, constr_type, tests)
    bestMetricesJointNEW3={}
    bestJointReferenceScoreNEW3={}
    for dnn in dnn_names:
        bestMetricesJointNEW3[dnn]=dataNEW3[dnn][best_jointNEW3]
        bestJointReferenceScoreNEW3[dnn]=objective(dataNEW3[dnn][best_jointNEW3]['l'], dataNEW3[dnn][best_jointNEW3]['e'], dataNEW3[dnn][best_jointNEW3]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
    top10_keysNEW3=[items[0] for items in top_10NEW3]
    #normalized_joint = [(items[1] - normmin) / (reference_best - normmin) for items in top_10]
    #print('normalized scores')
    #print(normalized_joint)
    best_scoresForEachNEW3={}
    scores_separateJointNEW3={}
    for dnn in dnn_names:
        reference_best=bestSeparateReferenceScore[dnn]
        scores2NEW3={}
        for keys in top10_keysNEW3:
            scores2NEW3[keys]=objective(dataNEW3[dnn][keys]['l'], dataNEW3[dnn][keys]['e'], dataNEW3[dnn][keys]['a'], coeff_lat, coeff_en, coeff_ar,obj_type)
        values_listNEW3 = [scores2NEW3[key] for key in top10_keysNEW3]
        #print(values_list)
        print('dnn:', dnn)
        print('best_Joint key:',top10_keysNEW3[0],'best_Joint value:',values_listNEW3[0])
        best_arch_scoreNEW3=scores2NEW3[best_jointNEW3]
        normalized_valuesNEW3 = [(x - normmin) / (reference_best - normmin) for x in values_listNEW3]
        scores_separateJointNEW3[dnn]=normalized_valuesNEW3
        best_scoresForEachNEW3[dnn]=(best_arch_scoreNEW3-normmin)/(reference_best - normmin)
        print('WHAT YOU NEEED: best_scoresForEach[dnn] (normalized score to reference)', best_scoresForEachNEW3[dnn])
        diffrNEW3=best_scoresForEachNEW3[dnn]-norm_score_bestSepForRef[dnn]
        diffrPersNEW3=diffrNEW3*100
        normalized_jointNEW3 = [(items[1] - normmin) / (reference_best - normmin) for items in top_10NEW3]
        #allSeparateScores[dnn]=normalized_joint
        percDiffNEW3[dnn]=diffrPersNEW3

    print('')
    print('Percentage difference in score (negative = joint is better)')
    print(percDiffNEW3)
    print('')
    print('best joint')
    
    print('A: Operating voltage, bits_per_cell, base_latency, crossbar_sizeX, crossbar_sizeY, Num tiles in a group sharing router, PEs in tile, crossbars in PE, glb_buffer_depth')
    partsNEW3 = best_jointNEW3.split('_')
    print(partsNEW3)
    # Regular expression to match both integers and floats
    number_patternNEW3 = re.compile(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$')
    # Extract numbers (integers or floats) from the split parts
    numbersNEW3 = [part for part in partsNEW3 if number_patternNEW3.match(part)]
    numbersNEW3[0]=numbersNEW3[0]+'V'
    numbersNEW3[1]=numbersNEW3[1]+'b'
    numbersNEW3[2]=numbersNEW3[2]+'s'
    numbersNEW3[8]=str(round(int(numbersNEW3[8])*2048/1024/1024/8))+'MB'
    #glb_buffer_depth=[int(1024*1024*x*8/2048) for x in [8,16,32,64,128,256]]
    print('Best joint architectures parameters',numbersNEW2)
    
    print('best joint')
    for dnn in dnn_names:
        print('   ', dnn)
        bestMetricesJointNEW3[dnn]['e']=bestMetricesJointNEW3[dnn]['e']*1e3 #in mJ
        bestMetricesJointNEW3[dnn]['l']=bestMetricesJointNEW3[dnn]['l']*1e9 #in ns
        bestMetricesJointNEW3[dnn]['a']=bestMetricesJointNEW3[dnn]['a']*1e2 #in mm2
        #bestMetricesJoint[dnn]['a_tpm']=bestMetricesJoint[dnn]['a_tpm']*ops[dnn]*bestMetricesJoint[dnn]['a'] #now TOPS (throughput)
        print('E =',bestMetricesJointNEW3[dnn]['e'],'mJ,  ', 'L =',bestMetricesJointNEW3[dnn]['l'],'ns,  ', 'A =',bestMetricesJointNEW3[dnn]['a'],'mm2')
        print('TOPS =',bestMetricesJointNEW3[dnn]['tp'], ',  TOPS/W =',bestMetricesJointNEW3[dnn]['a_tpw'], ',  TOPS/mm2 =',bestMetricesJointNEW3[dnn]['a_tpm'])
    
    # HERE PLOTTING STARTS!!!!!
    
    # Define separate lists for x and y values where x has fewer unique elements compared to y
    x_values = [1, 2, 3,4,5,6,8,9,10,11,12,13,15, 16,17,18, 19,20, 22,23, 24,25, 26,27]  # Unique x values
    
    percentage=[str(abs(round(percDiff[dnn])))+'%' for dnn in dnn_names]
    y_values = [
        
        allSeparateScores[dnn_names[0]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[0]],
        scores_separateJoint[dnn_names[0]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[0]], 
        scores_separateJointNEW2[dnn_names[0]], 
        scores_separateJointNEW3[dnn_names[0]], 
        
        
        allSeparateScores[dnn_names[1]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[1]],
        scores_separateJoint[dnn_names[1]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[1]], 
        scores_separateJointNEW2[dnn_names[1]], 
        scores_separateJointNEW3[dnn_names[1]], 
        
        allSeparateScores[dnn_names[2]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[2]],
        scores_separateJoint[dnn_names[2]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[2]], 
        scores_separateJointNEW2[dnn_names[2]], 
        scores_separateJointNEW3[dnn_names[2]], 
        
        allSeparateScores[dnn_names[3]],  # Corresponding y values for x = 2
        allSeparateScoresMAX[dnn_names[3]],
        scores_separateJoint[dnn_names[3]],  # Corresponding y values for x = 1
        scores_separateJointNEW[dnn_names[3]],
        scores_separateJointNEW2[dnn_names[3]],
        scores_separateJointNEW2[dnn_names[3]]
    ]
    
    #x_labels = ['Joint'+dnn_names[0], 'Separate'+dnn_names[0],'Joint'+dnn_names[1], 'Separate'+dnn_names[1],'Joint'+dnn_names[2], 'Separate'+dnn_names[2],'Joint'+dnn_names[3], 'Separate'+dnn_names[3]]
    x_labels = ['Separate', 'Max','Joint','JointNewSamp','JointNew2','JointNew4', 'Separate', 'Max','Joint','JointNewSamp','JointNew2','JointNew4', 'Separate', 'Max','Joint','JointNewSamp','JointNew2','JointNew4' ,'Separate', 'Max','Joint','JointNewSamp','JointNew2','JointNew4']
    
    # Define custom hex colors for each group
    #hex_colors = ['#76d3d7', '#76d3d7', '#a8e436', '#a8e436', '#a3cbef','#a3cbef', '#ffe822', '#ffe822']  # Blue, Green, and Orange shades
    #edge_colors=['#55979a', '#55979a', '#658922', '#658922', '#647e95','#647e95', '#9e901a', '#9e901a']
    hex_colors = ['#4361ee','#4361ee','#4361ee','#4361ee','#4361ee', '#4361ee', '#20d4f6', '#20d4f6','#20d4f6','#20d4f6','#20d4f6','#20d4f6','#d242ff','#d242ff','#d242ff','#d242ff', '#d242ff', '#d242ff', '#2a9d8f','#2a9d8f', '#2a9d8f','#2a9d8f', '#2a9d8f', '#2a9d8f']  # Blue, Green, and Orange shades
    edge_colors=['#32267d','#32267d', '#32267d','#32267d', '#32267d','#32267d','#2e7294','#2e7294','#2e7294', '#2e7294','#2e7294','#2e7294','#470080','#470080','#470080','#470080','#470080','#470080', '#264653', '#264653','#264653', '#264653', '#264653', '#264653']
    
    # Flatten the x and y lists to create a single list for plotting
    x_flat = [x for x, y_list in zip(x_values, y_values) for _ in y_list]  # Repeat each x according to the y elements
    y_flat = [y for y_list in y_values for y in y_list]  # Flatten all y values
    
    # Create the scatter plot
    
    fig1=plt.figure(figsize=(15, 6))
    
    # Plot each group with the specified hex colors and highlight the score with a star
    for i, (x, y_vals) in enumerate(zip(x_values, y_values)):
        if (i + 1) % 6 == 1:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='^',s=300, label=f'Group {x_labels[i]}')
        elif (i + 1) % 6 == 2:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='s',s=300, label=f'Group {x_labels[i]}')
        elif (i + 1) % 6 == 3:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='o',s=300, label=f'Group {x_labels[i]}')
        elif (i + 1) % 6 == 4:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='<',s=300, label=f'Group {x_labels[i]}')
        elif (i + 1) % 6 == 5:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='>',s=300, label=f'Group {x_labels[i]}')
        else:
            plt.scatter([x] * len(y_vals), y_vals, color=hex_colors[i],edgecolors=edge_colors[i], marker='v',s=300, label=f'Group {x_labels[i]}')
        # Highlight one dot per x value as a star with the same hex color
        plt.scatter(x, y_vals[0], color=hex_colors[i],edgecolors='#0E3C34', marker='*', s=750)  # 's' controls the size of the star
        # Add a custom legend text directly on the graph next to the first star point of each group
        
        # Add a custom legend text directly on the graph for even-numbered groups only
        #if (i + 1) % 2 == 0:  # Check if the group is even (i+1 because index i starts from 0)
        #    plt.text(x + 0.3, y_vals[0], f'{percentage[round(np.floor(i/2))]}', fontsize=10, color=hex_colors[i], va='center')
    
    
  
    # Adding custom legends
    custom_legends = [
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[0], markersize=10, label=dnn_names[0]),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[2], markersize=10, label=dnn_names[1]),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[4], markersize=10, label=dnn_names[2]),
        #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hex_colors[6], markersize=10, label=dnn_names[3]),
        #plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#0E3C34', markersize=15, label='Best architecture')
    ]
    
    plt.xlim(0, 28)  # Start from 0 and end at 4 on x-axis
    plt.ylim(0, ylimit) 
    #plt.xticks([])
    #plt.yticks([])
    #plt.legend(handles=custom_legends)
    # Adding labels and title
    #plt.xlabel('X-axis')
    #plt.ylabel('Normalized score')
    #plt.title('Scatter Plot with Separate X and Y Lists')
    plt.xticks(ticks=x_values, labels=x_labels)
    #plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.grid(axis='x')
    # Display the plot
    plt.tight_layout()
    plt.show()
    fig1.savefig(figname,dpi=2000)