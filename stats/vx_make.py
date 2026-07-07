# vx_make.py
# 작성일 2026.01.11
# VX_label 파일과, [S_ID, W_ID, "N_form", "N_label","V_form", "V_label", "EN_form", "EN_label", "J_form"]를 이용해서, 
# VX번호를 long type으로 만들어 주는 프로그램. 
# OUT_COLS_ADDED = ["Next_VX_No", "Next_VX_No_form","vx0_No", "vx0_form","vx0_order","vx_len","V_word_id","f_word_id"]

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path
from tqdm.auto import tqdm

# --- 설정 ---
W_ID = "word_id2"   #"word_id"
S_ID = "sen_id"     #"sent_id", "s_id"
FILE_ID = "file_id" #"docu_id"

KEEP_COLS = [
    "ID", FILE_ID, "category", S_ID, W_ID, "form/label",
    "N_form", "N_label", "V_form", "V_label",
    "EP_form", "EP_label", "EN_form", "EN_label", "J_form", "J_label",
    "EN_No"
]

DATA_COLS = [S_ID, W_ID, "N_form", "N_label",
               "V_form", "V_label", "EN_form", "EN_label", "J_form"]

VX_NO = "VX_No"
LABEL_COLS = [VX_NO, "EN_form",	"EN_label",	
              "N_form",	"N_label",	"V_form",	"V_label"]

OUT_COLS_ADDED = [
    "Next_VX_No", "Next_VX_No_form",
    "vx0_No", "vx0_form",
    "vx0_order",
    "vx_len",
    "V_word_id",
    "f_word_id"
]
#===========================
# --- 헬퍼 ---
def norm_empty(series):
    return series.fillna("").astype(str).str.strip()

def make_vx_form_two_row(
    en_form: str,
    en_j_form: str,
    n_form_next: str,
    v_form_next: str
) -> str:
    s = en_form
    if en_j_form:
        s += en_j_form
    if n_form_next:
        s += " " + n_form_next
    s += " " + v_form_next
    return s.strip()

def make_vx_form_three_row(
    en_form: str,
    n_form_mid: str,
    j_form_mid: str,
    v_form_last: str
) -> str:
    s = en_form + " " + n_form_mid
    if j_form_mid:
        s += j_form_mid
    s += " " + v_form_last
    return s.strip()

def load_label_map(label_csv: str | Path) -> pd.DataFrame:
    """
    Load mapping: (EN_label, EN_form) -> EN_No (base integer)
    label_csv must contain columns: EN_label, EN_form, EN_No
    """
    label_csv = Path(label_csv)
    lab = pd.read_csv(label_csv, low_memory=False)

    #0. 컬럼 확인
    missing = set(LABEL_COLS) - set(lab.columns) #없는 컬럼이 있는지 확인
    if missing:
        print(f"없는 컬럼: {missing} ")
        raise ValueError(f"{label_csv}에 다음 컬럼이 없습니다: {missing}")
    
    # label_df 준비 (필요 컬럼만, EN_No는 정수화)
    lab = lab[LABEL_COLS].copy()
    for c in LABEL_COLS:
        if c == VX_NO:
            continue
        lab[c] = norm_empty(lab[c])
    lab[VX_NO] = np.floor(pd.to_numeric(lab[VX_NO], errors="coerce")).astype("Int64")
    
    return lab

#========================================
# --- 핵심 로직 ---
# VX edge 생성 함수
def build_vx_edges_word_only(
    df_word: pd.DataFrame,
    lab: pd.DataFrame
) -> pd.DataFrame:
    
    df = df_word.copy()

    lab_emptyN = lab[(lab["N_form"]=="") & (lab["N_label"]=="")]
    lab_nonemptyN = lab[~((lab["N_form"]=="") & (lab["N_label"]==""))]

    df = df.sort_values([S_ID,W_ID]).reset_index(drop=True)
    g = df.groupby(S_ID, sort=False)

    # next1
    df["n1_wid"] = g[W_ID].shift(-1)
    df["n1_N_form"]   = norm_empty(g["N_form"].shift(-1))
    df["n1_N_label"]  = norm_empty(g["N_label"].shift(-1))
    df["n1_V_form"]   = norm_empty(g["V_form"].shift(-1))
    df["n1_V_label"]  = norm_empty(g["V_label"].shift(-1))

    # next2
    df["n2_wid"] = g[W_ID].shift(-2)
    df["n2_N_form"]   = norm_empty(g["N_form"].shift(-2))
    df["n2_N_label"]  = norm_empty(g["N_label"].shift(-2))
    df["n2_V_form"]   = norm_empty(g["V_form"].shift(-2))
    df["n2_V_label"]  = norm_empty(g["V_label"].shift(-2))

    # -------- 2-row --------
    cand2 = df[[
        S_ID,W_ID,"EN_form","EN_label","J_form",
        "n1_wid","n1_N_form","n1_N_label","n1_V_form","n1_V_label"
    ]].copy()

    cand2 = cand2.rename(columns={
        W_ID:"EN_wid",
        "n1_wid":"VX_wid",
        "n1_N_form":"N_form",
        "n1_N_label":"N_label",
        "n1_V_form":"V_form",
        "n1_V_label":"V_label",
        "J_form":"EN_J_form",
    })

    # label N empty
    cand2_empty = cand2[(cand2["N_form"]=="") & (cand2["N_label"]=="")]
    m2_empty = cand2_empty.merge(
        lab_emptyN,
        how="left",
        on=["EN_form","EN_label","N_form","N_label","V_form","V_label"]
    )

    # label N non-empty
    cand2_nonempty = cand2[~((cand2["N_form"]=="") & (cand2["N_label"]==""))]
    m2_nonempty = cand2_nonempty.merge(
        lab_nonemptyN,
        how="left",
        on=["EN_form","EN_label","N_form","N_label","V_form","V_label"]
    )

    m2 = pd.concat([m2_empty, m2_nonempty], ignore_index=True)
    m2 = m2[m2["VX_No"].notna()].copy()

    if len(m2):
        m2["VX_No"] = m2["VX_No"].astype(int)
        m2["VX_form"] = [
            make_vx_form_two_row(e, j, n, v)
            for e,j,n,v in zip(m2["EN_form"], m2["EN_J_form"], m2["N_form"], m2["V_form"])
        ]
        m2["match_len"] = 2

    # -------- 3-row (label N non-empty only) --------
    cand3 = df[[
        S_ID,W_ID,"EN_form","EN_label",
        "n1_wid","n1_N_form","n1_N_label",
        "n2_wid","n2_N_form","n2_N_label","n2_V_form","n2_V_label"
    ]].copy()

    cand3["J_form_mid"] = norm_empty(g["J_form"].shift(-1))

    cand3 = cand3.rename(columns={
        W_ID:"EN_wid",
        "n2_wid":"VX_wid",
        "n1_N_form":"N_form",
        "n1_N_label":"N_label",
        "n2_V_form":"V_form",
        "n2_V_label":"V_label",
        "n2_N_form":"N_form_last",
        "n2_N_label":"N_label_last",
    })

    cand3 = cand3[((cand3["N_form"]!="") | (cand3["N_label"]!=""))]
    cand3 = cand3[(cand3["N_form_last"]=="") & (cand3["N_label_last"]=="")]

    m3 = cand3.merge(
        lab_nonemptyN,
        how="left",
        on=["EN_form","EN_label","N_form","N_label","V_form","V_label"]
    )
    m3 = m3[m3["VX_No"].notna()].copy()

    if len(m3):
        m3["VX_No"] = m3["VX_No"].astype(int)
        m3["VX_form"] = [
            make_vx_form_three_row(e,n,j,v)
            for e,n,j,v in zip(m3["EN_form"], m3["N_form"], m3["J_form_mid"], m3["V_form"])
        ]
        m3["match_len"] = 3

    edges = pd.concat(
        [m2[[S_ID,"EN_wid","VX_wid","VX_No","VX_form","match_len"]] if len(m2) else pd.DataFrame(),
         m3[[S_ID,"EN_wid","VX_wid","VX_No","VX_form","match_len"]] if len(m3) else pd.DataFrame()],
        ignore_index=True
    )

    if len(edges):
        edges = edges.sort_values(
            [S_ID,"EN_wid","match_len","VX_wid"],
            ascending=[True,True,False,True]
        ).drop_duplicates(
            [S_ID,"EN_wid"], keep="first"
        ).reset_index(drop=True)

        edges["EN_wid"] = edges["EN_wid"].astype(int)
        edges["VX_wid"] = edges["VX_wid"].astype(int)

    return edges

# chain + vx_len 계산
def annotate_word_with_chains(
    df_word: pd.DataFrame,
    edges: pd.DataFrame
) -> pd.DataFrame:
    
    df = df_word.copy()
    df[S_ID] = norm_empty(df[S_ID])
    df[W_ID] = pd.to_numeric(df[W_ID], errors="coerce").astype("Int64")

    df = df.sort_values([S_ID,W_ID]).reset_index(drop=True)

    for c in OUT_COLS_ADDED:
        df[c] = pd.NA

    key_df = df[[S_ID, W_ID]].dropna()
    idx_map = {(sid, int(wid)): i for i, (sid, wid) in zip(key_df.index, key_df.to_numpy())}

    if edges is None or len(edges)==0:
        df["vx_len"] = 0
        return df

    edges = edges.copy()
    edges[S_ID] = norm_empty(edges[S_ID])
    edges["EN_wid"] = edges["EN_wid"].astype(int)
    edges["VX_wid"] = edges["VX_wid"].astype(int)
    edges["VX_No"] = edges["VX_No"].astype(int)

    sen_count = edges[S_ID].nunique(dropna=False) #진행 상황 표시를 위함
    for sen, e in tqdm(edges.groupby(S_ID, sort=False), total=sen_count, desc="Annotating VX chains"):
        out_map = {a:b for a,b in zip(e["EN_wid"], e["VX_wid"])}
        no_map  = {a:n for a,n in zip(e["EN_wid"], e["VX_No"])}
        form_map= {a:f for a,f in zip(e["EN_wid"], e["VX_form"])}

        incoming = set(out_map.values())
        roots = [n for n in out_map if n not in incoming] or list(out_map)

        vx_len_memo = {}
        final_memo = {}

        def compute(node):
            if node in vx_len_memo:
                return vx_len_memo[node]
            cur=node; seen=set(); length=0; last=node
            while cur in out_map and cur not in seen:
                seen.add(cur)
                cur = out_map[cur]
                length += 1
                last = cur
            vx_len_memo[node] = length
            final_memo[node] = last
            return length

        for n in set(out_map)|set(out_map.values()):
            compute(n)

        # EN rows
        for en, vx in out_map.items():
            k=(sen,en)
            if k in idx_map:
                i=idx_map[k]
                df.at[i,"Next_VX_No"]=no_map[en]
                df.at[i,"Next_VX_No_form"]=form_map[en]
                df.at[i,"vx_len"]=vx_len_memo.get(en,0)
                df.at[i,"f_word_id"]=final_memo.get(en,en)

        # VX rows
        for root in roots:
            cur=root; order=1; seen=set()
            while cur in out_map and cur not in seen:
                seen.add(cur)
                vx=out_map[cur]

                if (sen,vx) in idx_map:
                    j=idx_map[(sen,vx)]
                    df.at[j,"vx0_No"]=no_map[cur]
                    df.at[j,"vx0_form"]=form_map[cur]
                    df.at[j,"vx0_order"]=order
                    df.at[j,"V_word_id"]=root
                    df.at[j,"vx_len"]=vx_len_memo.get(vx,0)
                    df.at[j,"f_word_id"]=final_memo.get(vx,vx)

                if (sen,cur) in idx_map:
                    df.at[idx_map[(sen,cur)],"V_word_id"]=root

                cur=vx; order+=1

    df["vx_len"] = pd.to_numeric(df["vx_len"], errors="coerce").fillna(0).astype(int)
    for c in ["Next_VX_No","vx0_No","vx0_order","V_word_id","f_word_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df

#========================================
# 최종 호출
def make_annotate_vx(
    df_base: pd.DataFrame,
    label_csv: str | Path, 
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    입력:
        df_base  : 어절 단위 원본 말뭉치 DataFrame
        df_label : VX 라벨 정의된 csv 파일 경로.

    출력:
        df_out : VX 정보가 주석된 어절 단위 DataFrame
        edges  : EN → VX 연결 long-format DataFrame
    """
    #0.0 컬럼 확인 및 세팅
    missing = set(DATA_COLS) - set(df_base.columns) #없는 컬럼이 있는지 확인
    if missing:
        print(f"없는 컬럼: {missing} ")
        raise ValueError(f"DataFrame에 다음 컬럼이 없습니다: {missing}")
    
    #컬럼 세팅
    for c in DATA_COLS:
        if c == W_ID:
            continue
        df_base[c] = norm_empty(df_base[c]) if c in df_base.columns else ""
    df_base[W_ID] = pd.to_numeric(df_base[W_ID], errors="coerce").astype("Int64")
    print(f"컬럼을 세팅하였습니다. {DATA_COLS} ")

    #0.1 label 불러오기. 
    lab = load_label_map(label_csv) #label.csv읽어오기
    print(f"label을 읽어왔습니다. {label_csv} ")

    #1. 실행
    print(f"edges를 만듭니다.")
    edges = build_vx_edges_word_only(df_base, lab)

    print(f"vx 컬럼을 만듭니다.")
    df_out = annotate_word_with_chains(df_base, edges)

    return df_out, edges