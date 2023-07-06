import re
import pandas as pd
import numpy as np
import os
import subprocess
from collections import defaultdict

# Authors: 
MODEL_DIR = '/Documents/GitHub/deepddi2/'
INPUT_PATH = './DDI_input.txt'
OUTPUT_DIR = './output'
OUTPUT_TXT = 'output/Final_annotated_DDI_result.txt'
SIGNIFICANCE = 0.8
DFI_INPUT_DRUGS = []
DDI_OTHER_DRUGS = []
DFI_FOOD_LIST = []

food_comp = pd.read_csv('./database/food_compounds_lookup.csv')
# pd.DataFrame({'Name': food.orig_food_common_name\
#                           .str.replace('(', '').str.replace(')', '')\
#                           .str.split()\
#                           .str[0].unique()}
#             ).to_csv('./database/food.csv', index = None)

food_loc = food_comp.orig_food_common_name\
            .str.replace('(', '')\
            .str.replace(')', '')\
            .str.split().str[0]
comp_loc = food_comp.name.str.lower()

food2comp = defaultdict(set)
comp2food = defaultdict(set)
for f,c in zip(food_loc, comp_loc):
    food2comp[f].add(c)
    comp2food[c].add(f)
    

# 查找文本是否有匹配的
def regex_search(desc, pools):
    # assume desc is lowercased
    out = []
    for elem in pools:
        pattern = elem.strip().lower()
        if re.search(pattern, desc):
            out.append(elem)
    return out

# 生成first_line : Drug D|vitamin c        有文本记录的药物名称
# 生成second_line : lemon|cholesterol      其他药物
def ingest_input(input_json, interaction_type, input_fp = INPUT_PATH,
                 compounds_path = './database/drug_info_combined.csv',
                 food_path = './database/food.csv'):
    assert interaction_type.lower() in ['ddi', 'dfi'], 'API not supported'
    first_line = []
    second_line = []
    # handle ddi
    if interaction_type.lower() == 'ddi':
        # Name,Drug name,Smiles
        # Bivalirudin,DB00006,CC[C@H](C)[C@H](NC(=O)[C@H]…………
        drug_pools = pd.read_csv(compounds_path).Name.str.lower() # 所有有药物信息(编号、smiles）的药物的名字
        
        cur_desc = input_json['current_drug']['drug_desc'].lower() # 存储当前行输入药物信息的文本描述
        drug_title = input_json['current_drug']['drug_title'] # 存储当前行输入药物信息的药品名称
        
        drug_search = regex_search(cur_desc.lower(), drug_pools) # 从文本描述里找有哪些药物
        # 若为空 表示drug title的药物的文本描述里没有已知药物信息的药物
        assert drug_search, ('Drug: %s not Found' % drug_title) # assert语句用于确保drug_search列表不为空。如果drug_search为空（即没有找到匹配的药物），assert语句将引发一个异常，其中包含错误消息"Drug: %s not Found"，其中的%s将被具体的drug_title值替换。在描述文本中找不到指定的药物
            
        first_line += [drug_title+'|'+i for i in drug_search]  # 药物|药物成分
        for drug in input_json['other_drug']:
            DDI_OTHER_DRUGS.append(drug['drug_title'].lower())
            other_desc = drug['drug_desc'].lower()
            other_search = regex_search(other_desc, drug_pools)
            
            if not other_search:
                print('Drug: %s not Found' % drug['drug_title'])
                continue
            second_line += [drug['drug_title']+'|'+ i for i in other_search]
    # handle dfi
    else:
        drug_pools = pd.read_csv(compounds_path).Name.str.lower()
        food_pools = pd.read_csv(food_path).Name
        for drug in input_json['drug_list']:
            DFI_INPUT_DRUGS.append(drug['drug_title'].lower())
            drug_search = regex_search(drug['drug_desc'].lower(), drug_pools)
#             print(drug['drug_desc'], drug_search)
            if not drug_search:
                print('Drug: %s not Found' % drug['drug_title'])
                continue
            first_line += [drug['drug_title'] + '|' + i for i in drug_search]
        
        for food in input_json['food_list']:
            food_search = food2comp[food.lower()] 
            if not food_search:
                print('Food: %s not Found' % food)
                continue
            DFI_FOOD_LIST.append(food)
            second_line += [food + '|' + i for i in food_search]
            
    # TODO handle not-found case           
    if os.path.exists(input_fp):
        os.remove(input_fp)
    
    with open(input_fp, 'w') as fw:
        fr = '\t'.join(first_line) + '\n'
        sr = '\t'.join(second_line) + '\n'
        fw.write(fr)
        fw.write(sr)


def run(input_json,interaction_type,thres=SIGNIFICANCE):
    # INPUT: 
    #   input_json: the json file of input info
    #   interactioin_type: 'DFI' or 'DDI'
    # execute & make sure it runs linearly
    ingest_input(input_json, interaction_type) # 调用 ingest_input 函数，将输入的 JSON 文件解析为文本文件，并保存为 DDI_input.txt 文件

    # 这个cmd的作用是：
    cmd = ('/root/miniconda3/bin/python3.8 run_DeepDDI.py -i %s -o %s -t %s'%('DDI_input.txt', 'output',str(interaction_type))).split()
    try:
        subprocess.run(cmd)
        if interaction_type=='DFI':
            return collect_food_output(thres)
        else:
            return collect_drug_output(thres)
    except AssertionError:
        return None

# cmd结果：(output/Final_DDI_result.txt)
# Prescription    Drug_pair    Interaction_type    Sentence    DDI_prob    DDI_prob_std    Confidence_DDI    Side effects (left)    Side effects (right)    Similar approved drugs (left)    Similar approved drugs (right)    drug1    drug2
# prescription1   drug1_drug2  interaction_type1  sentence1   0.8          0.1             1                  side_effect1(20.0%)      side_effect2(15.5%)          similar_drug1;similar_drug2  similar_drug3;similar_drug4  drug1    drug2
def collect_drug_output(thres = SIGNIFICANCE, out_txt = OUTPUT_TXT):
    #使用 Pandas 的 read_csv 函数读取外部命令生成的输出文件，并提取其中的部分列，包括药物1、药物2、DDI概率、DDI概率标准差、DDI置信度、句子、药物1的副作用、药物2的副作用
    res = pd.read_csv(out_txt,
                      sep='\t', 
                      header=0)[['drug1', 'drug2',
                                          'DDI_prob', 'DDI_prob_std',
                                          'Confidence_DDI', 'Sentence',
                                          'Side effects (left)',
                                          'Side effects (right)']]
    #通过筛选条件 Confidence_DDI == 1 和 DDI_prob >= thres，从结果中选出置信度为1且DDI概率大于等于阈值的记录，保存到 temp 变量中
    temp = res.loc[(res.Confidence_DDI == 1) &\
                   (res.DDI_prob >= thres)]
    
#     print(temp) 如果设置了 DDI_OTHER_DRUGS，则根据该列表中的药物名字，进一步筛选 temp 中的记录，保留药物1或药物2名称中包含指定字符串的记录
    if DDI_OTHER_DRUGS:
        temp = temp.loc[(res.drug1.str.contains(r'|'.join(DDI_OTHER_DRUGS))) |\
                        (res.drug2.str.contains(r'|'.join(DDI_OTHER_DRUGS)))]
        
#     if DFI_INPUT_DRUGS:
#         temp = temp.loc[(res.drug1.str.contains(r'|'.join(DFI_INPUT_DRUGS))) |\
#                         (res.drug2.str.contains(r'|'.join(DFI_INPUT_DRUGS)))]
    processed_other_drugs = []
    out = {}
    out['drug_interactions'] = []
    out['cur_drug_side_effect'] = []
    
     # 遍历每条记录，提取药物相互作用信息和当前药物副作用信息
    for line in temp.iterrows():
        inner_out = {}
        
        row = line[1].values
        other_drug = row[1]
        cur_drug = row[0]
#         print('-->', row)

 # 提取药物1和药物2的名称
        drug1=re.findall('(.*)\(.*$',row[0])[0]
        drug2=re.findall('(.*)\(.*$',row[1])[0]
        
# 如果设置了 DDI_OTHER_DRUGS，交换药物1和药物2的位置
        if DDI_OTHER_DRUGS:
            if drug2 in DDI_OTHER_DRUGS:
                cur_drug, other_drug = drug1, drug2
                
            elif drug1 in DDI_OTHER_DRUGS:
                cur_drug, other_drug = drug2, drug1
        
 # 如果药物2已经被处理过了，更新相互作用的概率和描述信息
        if other_drug in processed_other_drugs:
            for i in out['drug_interactions']:
                if (other_drug == i['other_drug_name']) and (row[2] > i['probability']):
                    i['probability'] = row[2]
                    i['interaction_desc'] = row[5]
# 否则，将药物1和药物2之间的相互作用信息添加到 out 字典对象中
        else:        
            inner_out['probability'] = row[2]
            inner_out['other_drug_name'] = other_drug
            inner_out['interaction_desc'] = row[5]
            out['drug_interactions'].append(inner_out)
            processed_other_drugs.append(other_drug)

 # 提取当前药物的副作用信息
#         print(type(row[6]), type(row[7]), row[0], cur_drug)
        side_effect_str = None
        if (cur_drug.lower() in row[0].lower()) \
                    and (isinstance(row[6], str)): # left
            side_effect_str = row[6]
            
        elif (cur_drug.lower() in row[1].lower()) \
                    and (isinstance(row[7], str)): # right
            side_effect_str = row[7]

# 如果副作用信息不为空，提取其中的副作用名称和概率
        if side_effect_str:
#             print(side_effect_str)
            pools = re.split(';|\s', side_effect_str)
#             print(pools)
            for se in pools:
                cur_drug_side_effect = {}
                name, num = re.findall('^\w+',se)[0], re.findall('\((.*)\)',se)[0]
                cur_drug_side_effect['side_effect_name'] = name
                cur_drug_side_effect['probability'] = num
                out['cur_drug_side_effect'].append(cur_drug_side_effect)
    return out

def collect_food_output(thres = SIGNIFICANCE, out_txt = OUTPUT_TXT):
    res = pd.read_csv(out_txt,
                      sep='\t', 
                      header=0)[['drug1', 'drug2',
                                          'DDI_prob', 'DDI_prob_std',
                                          'Confidence_DDI', 'Sentence',
                                          'Side effects (left)',
                                          'Side effects (right)']]
    
    temp = res.loc[(res.Confidence_DDI == 1) &\
                   (res.DDI_prob >= thres)]
    
#     if DDI_OTHER_DRUGS:
#         temp = temp.loc[(res.drug1.str.contains(r'|'.join(DDI_OTHER_DRUGS))) |\
#                         (res.drug2.str.contains(r'|'.join(DDI_OTHER_DRUGS)))]
        
    if DFI_INPUT_DRUGS:
        temp = temp.loc[(res.drug1.str.contains(r'|'.join(DFI_INPUT_DRUGS))) |\
                        (res.drug2.str.contains(r'|'.join(DFI_INPUT_DRUGS)))]
    
    out = []
    temp['drug_name']=temp['drug1'].apply(lambda x: re.findall('^[^\(]+',x)[0])
    for i in DFI_FOOD_LIST:
        each_food={}
        drug_interactions=[]
        food_i=temp.loc[temp['drug2'].str.contains(i.lower())]
        find_most=food_i[food_i.groupby('drug_name')['DDI_prob'].transform(max) == food_i['DDI_prob']]
        for row in find_most.iterrows():
            row=row[1]
            each_row={}
            each_row['other_drug_name']=row['drug_name']
            each_row['interaction_desc']=row['Sentence']
            each_row['probability'] = str(float(row['DDI_prob'])*100)[:4]+'%'
            drug_interactions.append(each_row)
        each_food['food_name']=i
        each_food['drug_interactions']=drug_interactions
        out.append(each_food)
    return out

# Example of calling DFI API
dfi_sample_input = {'drug_list': [{'drug_title': 'Drug C', 'drug_desc': '?   '}, {'drug_title': 'Drug D', 'drug_desc': '? Vitamin C '}],
 'food_list': ['lemon']}

# Example of calling DDI API
ddi_sample_input = {
  'current_drug': {
    'drug_title': 'Current Drug',
    'drug_desc': 'Vitamin C  Ritonavir'
  },
  'other_drug': [
    {
      'drug_title': 'Other Drug A',
      'drug_desc': ' cool  Vitamin A '
    },
    {
      'drug_title': 'Other Drug B',
      'drug_desc': ' very good Riboflavin Acetaminophen'
    },
    {
      'drug_title': 'Other Drug C',
      'drug_desc': ' very good Formoterol'
    }
  ]
}



# 预测的是 当前药物 和其它几种药物 相互作用的DDI类型和描述
# ALL you need to call is func 'run(input_json,type)'
print(run(ddi_sample_input,'DDI'))

print("over")
# print(run(dfi_sample_input,'DFI'))

    