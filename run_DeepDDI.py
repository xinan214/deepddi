import argparse
import os
import shutil
import logging
import time

from deepddi import DeepDDI
from deepddi import Severity
from deepddi import preprocessing
from deepddi import result_processing

# 生成文件部分列，包括药物1、药物2、DDI概率、DDI概率标准差、DDI置信度、句子、药物1的副作用、药物2的副作用

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-o', '--output_dir', required=True, help="Output directory")
    parser.add_argument('-i', '--input_file', required=True, help="Input file")
    parser.add_argument('-t', '--DDI_type', default='DDI', help="Enter 'DDI' or 'DFI'")
    # parser.add_argument('-m', '--ddi_trained_model', required=True, help = "'drugbnak' or 'manual'")
    
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    options = parser.parse_args()
    raw_input_file = options.input_file #./DDI_input.txt:药物|成分1
    output_dir = options.output_dir # output
    input_type = options.DDI_type.lower() # DDI
    if input_type not in ['ddi','dfi']:
        print("'Please enter 'DDI' or 'DFI' and try again")
        exit()
    
    drug_dir = './data/DrugBank5.0_Approved_drugs/'
    pca_model = './data/PCA_tanimoto_model_50.pkl'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    drug_information_file = './data/Approved_drug_Information.txt'
    drug_enzyme_information_file = './data/Approved_drug_enzyme_Information.txt'   
    side_effect_information_file = './data/Drug_Side_Effect.txt'
    
    drug_list = []
    with open('./data/DrugList.txt', 'r') as fp:
        for line in fp:
            drug_list.append(line.strip())

    input_file = '%s/DDI_input.txt' % (output_dir)
    if input_type=='ddi':
        parsed_input = preprocessing.parse_drug_input(raw_input_file)

    elif input_type=='dfi':
        parsed_input = preprocessing.parse_food_input(raw_input_file)

    # Prescription1	DrugA	SMILES_A	DrugB	SMILES_B
    # Prescription2	DrugC	SMILES_C	DrugD	SMILES_D
    preprocessing.parse_DDI_input_file('data/parsed_input.txt', input_file)
    
    known_drug_similarity_file = './data/drug_similarity.csv' #2386种药物的相似度矩阵
    similarity_profile = known_drug_similarity_file


    similarity_profile = '%s/similarity_profile.csv' % output_dir
    pca_similarity_profile = '%s/PCA_transformed_similarity_profile.csv' % output_dir
    pca_profile_file = '%s/PCA_transformed_similarity_profile.csv' % output_dir
    print ('Calculate structure similarity profile')
    #drug_dir:'./data/DrugBank5.0_Approved_drugs/' input_file:'output/DDI_input.txt'
    # similarity_profile :'output/similarity_profile.csv' drug_list:  ./data/DrugList.txt
    preprocessing.calculate_structure_similarity(drug_dir, input_file, similarity_profile, drug_list)

    # pca_1 …… pca_50
    preprocessing.calculate_pca(similarity_profile, pca_similarity_profile, pca_model) #降维到50 输出文件：output/PCA_transformed_similarity_profile.csv

    print ('Combine structural similarity profile')

    # 0	current drug(vitamin c)	[H][C@@]1(OC(=O)C(O)=C1O)[C@@H](O)CO	other drug a(vitamin a)	C\C(=C/CO)\C=C\C=C(/C)\C=C\C1=C(C)CCCC1(C)C
    # ,PC_1,PC_2,……,PC_50
    # 0_current drug(vitamin c)_other drug a(vitamin a) 1_PC_1 1_PC_50 2_PC_1 2_PC_50
    pca_df = preprocessing.generate_input_profile(input_file, pca_similarity_profile)

    # threshold = 0.5
    model1_threshold = 0.4

    # model processing
    print('model running')

    model1_dir = output_dir

    ddi_trained_model = './data/models/ddi_model.json'
    ddi_trained_model_weight = './data/models/ddi_model.h5'
    DDI_sentence_information_file = './data/Type_information/Interaction_information_model1.csv'
    binarizer_file = './data/multilabelbinarizer.pkl'
    known_DDI_file = './data/DrugBank_known_ddi.txt'

    output_file = '%s/DDI_result.txt' % (model1_dir)
    ddi_output_file = '%s/Final_DDI_result.txt' % (model1_dir)   
    annotation_output_file = '%s/Final_annotated_DDI_result.txt' % (model1_dir)
    model_type = 'model1'

    # output_file='output/DDI_result.txt' pca_df是输入数据
    # 输出：Drug pair	Predicted class	Score	STD
    # 输入文件格式：0_other drug a(vitamin a)_current drug(vitamin c)	113	0.598348	0.25049192
    DeepDDI.predict_DDI(output_file, pca_df, ddi_trained_model, ddi_trained_model_weight, model1_threshold, binarizer_file, model_type)

    # Prescription    Drug_pair    DDI_type    Sentence                   Score    STD
    # 0   drug1_drug2  DDI_type1   Drug1 may interact with Drug2  0.8      0.1
    result_processing.summarize_prediction_outcome(output_file, ddi_output_file, DDI_sentence_information_file)

    # Prescription    Drug_pair    Interaction_type    Sentence    DDI_prob    DDI_prob_std    Confidence_DDI    Side effects (left)    Side effects (right)    Similar approved drugs (left)    Similar approved drugs (right)    drug1    drug2
    # prescription1   drug1_drug2  interaction_type1  sentence1   0.8          0.1             1                  side_effect1(20.0%)      side_effect2(15.5%)          similar_drug1;similar_drug2  similar_drug3;similar_drug4  drug1    drug2
    result_processing.annotate_DDI_results(ddi_output_file, drug_information_file, drug_enzyme_information_file,
    similarity_profile, known_DDI_file, annotation_output_file, side_effect_information_file, model1_threshold, 0.7)    

    logging.info(time.strftime("Elapsed time %H:%M:%S", time.gmtime(time.time() - start)))
