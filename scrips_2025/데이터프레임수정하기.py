# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 11:30:53 2023

@author: yu
"""

import os
import pandas as pd
import numpy as np

from traceback import format_exc
from datetime import datetime
#%% 0.파일 읽어오기, MP전체 저장파일

time = []#걸린 시간 체크
start_t = datetime.now() #시간 체크
time.append(datetime.now() - start_t) #시간간격 저장
print(f"0. 시작합니다. {datetime.now()}")
#파일 경로
dir_path = 'C:/Users/yu/OneDrive/★copus/☆모두의말뭉치/'#'C:/Users/yu/OneDrive/★copus/★21세기 세종계획/2023/' # #
folder_name = '국립국어원 일상대화 말뭉치2020,2021/' #'국립국어원 구어 말뭉치(버전 1.2)/' #'2023/' # #
input_file_name =  '' #"세종_문어_형태분석_말뭉치"

folder_path = dir_path + folder_name
sen_ids = ["sen_id"] # 파일에 있는 문장번호까지의 id들 넣기

if input_file_name == "": #폴더만 지정된 경우. #폴더 안에 sen폴더와 word폴더로 csv파일 들어 있음.
    word_folder = 'word' # 폴더 안에 _word_ver3.csv 파일 넣어줘야 함. 
    sen_folder = 'sen'   # _sen.csv 파일 넣어줘야 함. 
    input_file_name = folder_name.split('/')[0]        #'NIKL_SPOKEN_v1.2_15' #'NIKL_DIALOGUE_2020_v1.3'
    case = 1
else:
    case = 2    
#%% 1. 파일 읽어오기
### 1.1 전체 파일-word_ver3 읽기
print('   1.1 전체 파일-word_ver4 읽기') 
if case ==1 or case == 2:
    print('1. 전체 파일-word_ver4 읽기')
    with open(folder_path + input_file_name + '_word_ver4.csv', "r", encoding = "UTF-8") as f:
            print(f"   {input_file_name}_word_ver4.csv를 읽어옵니다.")
            df = pd.read_csv(f)
            #'Unnamed:'를 포함하는 columns 삭제
            cols_to_drop = [col for col in df.columns if 'Unnamed:' in col]
            df = df.drop(columns=cols_to_drop)
#%%     
vx_counts2 = df.groupby(['vx1', 'vx1_No']).size().reset_index(name='Count')
en_counts2 = df.groupby(['EN_form', 'EN_label', 'EN_No']).size().reset_index(name='Count')
'''
vx_counts = pd.merge(vx_counts, vx_counts2, how = 'outer', on=['vx1', 'vx1_No'])
en_counts = pd.merge(en_counts, en_counts2, how = 'outer', on=['EN_form', 'EN_label', 'EN_No'])
en_counts.to_csv('en_conver.csv')
vx_counts.to_csv('vx_conver.csv')
'''
#%%       
# 데이터프레임 예시
df = pd.DataFrame({
    'Name': ['John', 'Alice', 'Bob', 'Charlie'],
    'Age': [25, 28, 22, 30],
    'City': ['New York', 'Paris', 'London', 'Tokyo']
})

# 사용자 입력 받기
search_column = input("조회할 열을 입력하세요: ")
search_value = input("조회할 값을 입력하세요: ")
update_column = input("수정할 열을 입력하세요: ")
update_value = input("수정할 값을 입력하세요: ")

# 데이터프레임 조회 및 조건 설정
mask = df[search_column] == search_value
filtered_df = df[mask]

# 조회된 행 확인 및 수정 여부 확인
print("조회된 행:")
print(filtered_df)

if filtered_df.empty:
    print("조회된 행이 없습니다.")
else:
    confirm = input("수정할까요? (Y/N): ")
    if confirm.lower() == 'y':
        # 조회된 행의 다른 열 값을 수정
        filtered_df.loc[:, update_column] = update_value

        # 원본 데이터프레임에 수정된 행 복사 및 업데이트
        df.update(filtered_df)

        # 변경된 데이터프레임 확인
        print("\n수정된 데이터프레임:")
        print(df)
    else:
        print("수정이 취소되었습니다.")
