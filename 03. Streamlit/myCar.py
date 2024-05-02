import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
from pycaret.regression import *
import random
from datetime import datetime
warnings.filterwarnings('ignore')



##################################### Session state ##############################################

if 'carName' not in st.session_state :
    st.session_state['carName'] = ''

if 'name' not in st.session_state :
    st.session_state['name'] = 'normal'

if 'year' not in st.session_state :
    st.session_state['year'] = 20

if 'fuel' not in st.session_state :
    st.session_state['fuel'] = 'gasoline'

if 'color' not in st.session_state :
    st.session_state['color'] = 'white'

if 'distance' not in st.session_state :
    st.session_state['distance'] = 0

    # Options(10)
    # smartkey, auto_mirror, power_window
    # elec_parking_break, AUX
    # around_view, HUD, driver_memory_seat
    # leather_seat(leahter_seat 오타 주의)
    # fullauto_aircon

if 'smartkey' not in st.session_state :
    st.session_state['smartkey'] = 0

if 'auto_mirror' not in st.session_state :
    st.session_state['auto_mirror'] = 0

if 'power_window' not in st.session_state :
    st.session_state['power_window'] = 0

if 'elec_parking_break' not in st.session_state :
    st.session_state['elec_parking_break'] = 0

if 'AUX' not in st.session_state :
    st.session_state['AUX'] = 0

if 'around_view' not in st.session_state :
    st.session_state['around_view'] = 0

if 'HUD' not in st.session_state :
    st.session_state['HUD'] = 0

if 'driver_memory_seat' not in st.session_state :
    st.session_state['driver_memory_seat'] = 0

if 'leahter_seat' not in st.session_state :
    st.session_state['leahter_seat'] = 0

if 'fullauto_aircon' not in st.session_state :
    st.session_state['fullauto_aircon'] = 0


###################################### Load Models & columns #############################################

columns_grandeur = np.load('./columns/columns_grandeur.npy', allow_pickle=True)
columns_k5 = np.load('./columns/columns_k5.npy', allow_pickle=True)
columns_morning = np.load('./columns/columns_morning.npy', allow_pickle=True)
columns_tucson = np.load('./columns/columns_tucson.npy', allow_pickle=True)

colors_grandeur = np.load('./colors/colors_grandeur.npy', allow_pickle=True)
colors_k5 = np.load('./colors/colors_k5.npy', allow_pickle=True)
colors_morning = np.load('./colors/colors_morning.npy', allow_pickle=True)
colors_tucson = np.load('./colors/colors_tucson.npy', allow_pickle=True)




model_grandeur = load_model('./models/grandeur_auto')
model_k5 = load_model('./models/k5_auto')
model_morning = load_model('./models/morning_auto')
model_tucson = load_model('./models/tucson_auto')

model_margin = joblib.load('./models/margin_2ndPoly.pkl')
model_poly = joblib.load('./models/2ndPoly.pkl')


####################################### Define Function ################################################



def createNewDF(columns_carName) :
    df = pd.DataFrame(columns=columns_carName[:-1])
    df.loc[0] = [0] * len(df.columns)
    return df
    


def inputInfo(df, name, year, distance, fuel, color, selected_options) :

    # Options(10)
    # smartkey, auto_mirror, power_window
    # elec_parking_break, AUX
    # around_view, HUD, driver_memory_seat
    # leather_seat(leahter_seat 오타 주의)
    # fullauto_aircon

    name = 'name_' + name
    fuel = 'fuel_' + fuel
    color = 'color_' + color
    df[name] = 1
    df['year'] = year
    df['distance'] = distance
    df[fuel] = 1
    df[color] = 1
    df[selected_options] = 1
    
    # 나머지 옵션 랜덤 배정 (feature importance 굉장히 낮은 옵션)
    random.seed(int(datetime.now().microsecond))
    options_len = len(df.loc[:, 'sunroof':].columns)
    tmp_rand = [random.randint(0, 1) for _ in range(options_len - len(selected_options))]
    df.loc[:, [i for i in list(df.loc[:, 'sunroof':].columns) if i not in selected_options]] = tmp_rand


    return df

# 마진율이 포함된 가격 예측
def predict(model, df) :
    predict = predict_model(model, df)
    return int(predict['prediction_label'][0])

# 마진율 예측 함수
def calcMargin(price) :
    price_poly = model_poly.transform(np.array(price).reshape(-1, 1))
    margin = np.round(model_margin.predict(price_poly), 2)[0]
    return margin


# 마진율을 고려한 최종 가격 예측
def predictPrice(price, margin) :
    return int(price - (margin / 100 * price))


################################### Main Page ##################################################


col1, col2, col3 = st.columns([1, 3, 1])
with col1 :
    
    ''


with col2 :
    ''
    ''
    
    st.markdown("""
        <style>
        button[title="View fullscreen"]{
        visibility: hidden;
        }
        .custom-font {
        font-size:50px !important;
        text-align : center;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<p class="custom-font">내차팔아보아요~</p>', unsafe_allow_html=True)

    

with col3 :
    ''


################################### Side Bar ##################################################

grandeurs = ['그랜저', '그랜저HG', '그랜저IG', '그랜저TG', '더 올 뉴 그랜저', '더 럭셔리 그랜저', '더 뉴 그랜저']
mapper_grandeur = {'그랜저' : 'normal', 
                   '그랜저HG' : 'HG', 
                   '그랜저IG' : 'IG',
                   '그랜저TG' : 'TG', 
                   '더 올 뉴 그랜저' : 'the_all_new', 
                   '더 럭셔리 그랜저' : 'the_luxury', 
                   '더 뉴 그랜저' : 'the_new'}

mapper_grandeur_inv = {v : k for (k, v) in mapper_grandeur.items()}



k5s = ['K5', '2세대', '3세대', '더 뉴 K5', '더 뉴 K5 2세대', '더 뉴 K5 3세대']
mapper_k5 = {'K5' : 'normal', 
                   '2세대' : '2nd', 
                   '3세대' : '3rd', 
                   '더 뉴 K5' : 'the_new', 
                   '더 뉴 K5 2세대' : 'the_new_2nd', 
                   '더 뉴 K5 3세대' : 'the_new_3rd'}
mapper_k5_inv = {v : k for (k, v) in mapper_k5.items()}

mornings = ['모닝', '올 뉴 모닝', '올 뉴 모닝밴', '뉴 모닝 L', '뉴 모닝 LX', '뉴 모닝 SLX', 
             '더 뉴 모닝', '더 뉴 모닝밴']
mapper_morning = {'모닝' : 'normal', 
                   '올 뉴 모닝' : 'all_new', 
                   '올 뉴 모닝밴' : 'all_new_van', 
                   '뉴 모닝 L' : 'new_L', 
                   '뉴 모닝 LX' : 'new_LX', 
                   '뉴 모닝 SLX' : 'new_SLX', 
                   '더 뉴 모닝' : 'the_new', 
                   '더 뉴 모닝밴' : 'the_new_van'}
mapper_morning_inv = {v : k for (k, v) in mapper_morning.items()}

tucsons = ['올 뉴 투싼', '투산 ix', '뉴 투싼 ix']
mapper_tucson = {'올 뉴 투싼' : 'all_new', 
                 '투산 ix' : 'ix', 
                 '뉴 투싼 ix' : 'new_ix'}
mapper_tucson_inv = {v : k for (k, v) in mapper_tucson.items()}







with st.sidebar.form(key='myCar', clear_on_submit=False) :
    st.header('차종 선택')

    carName = st.radio('차종', ['그랜저', 'K5', '투싼', '모닝'])



    if st.form_submit_button('Submit') :
        st.session_state['carName'] = carName
        
        
        st.experimental_rerun()



with st.sidebar.form(key='myCar2', clear_on_submit=False) :
    st.header('세부 사항 선택')

    if st.session_state['carName'] == '그랜저' :
        
        name = st.selectbox('모델', grandeurs)
        name = mapper_grandeur[name]
        color = st.radio('색상', list(colors_grandeur))
        fuel = st.radio('연료', ['휘발유', '경유', 'LPG', '하이브리드(휘발유)'])

    elif st.session_state['carName'] == 'K5' :
        
        name = st.selectbox('모델', k5s)
        name = mapper_k5[name]
        color = st.radio('색상', list(colors_k5))
        fuel = st.radio('연료', ['휘발유', '경유', 'LPG', '하이브리드(휘발유)'])

    elif st.session_state['carName'] == '투싼' :
        
        name = st.selectbox('모델', tucsons)
        name = mapper_tucson[name]
        color = st.radio('색상', list(colors_tucson))
        fuel = st.radio('연료', ['휘발유', '경유'])

    elif st.session_state['carName'] == '모닝' :
        
        name = st.selectbox('모델', mornings)
        name = mapper_morning[name]
        color = st.radio('색상', list(colors_morning))
        fuel = st.radio('연료', ['휘발유', 'LPG'])


    year = st.slider('연식', 9, 24)

    distance = st.number_input('주행거리', 0, 300000)


    smartkey = st.checkbox('스마트키', False)
    auto_mirror = st.checkbox('전동 접이식 미러', False)
    power_window = st.checkbox('파워 윈도우', False)
    elec_parking_break = st.checkbox('전자식 파킹 브레이크', False)
    AUX = st.checkbox('AUX', False)
    around_view = st.checkbox('어라운드 뷰', False)
    HUD = st.checkbox('HUD', False)
    driver_memory_seat = st.checkbox('운전석 메모리시트', False)
    leahter_seat = st.checkbox('가죽시트', False)
    fullauto_aircon = st.checkbox('풀오토 에어컨', False)



    if st.form_submit_button('Submit') :
        st.session_state['name'] = name
        st.session_state['year'] = year
        st.session_state['fuel'] = fuel
        st.session_state['distance'] = distance
        st.session_state['color'] = color
        st.session_state['smartkey'] = smartkey
        st.session_state['auto_mirror'] = auto_mirror
        st.session_state['power_window'] = power_window
        st.session_state['elec_parking_break'] = elec_parking_break
        st.session_state['AUX'] = AUX
        st.session_state['around_view'] = around_view
        st.session_state['HUD'] = HUD
        st.session_state['driver_memory_seat'] = driver_memory_seat
        st.session_state['leahter_seat'] = leahter_seat
        st.session_state['fullauto_aircon'] = fullauto_aircon
       
        
        st.experimental_rerun()
        

option_dict = {'smartkey' : st.session_state['smartkey'],
               'auto_mirror' : st.session_state['auto_mirror'],
               'power_window' : st.session_state['power_window'],
               'elec_parking_break' : st.session_state['elec_parking_break'],
               'AUX' : st.session_state['AUX'],
               'around_view' : st.session_state['around_view'],
               'HUD' : st.session_state['HUD'],
               'driver_memory_seat' : st.session_state['driver_memory_seat'],
               'leahter_seat' : st.session_state['leahter_seat'],
               'fullauto_aircon' : st.session_state['fullauto_aircon']}


################################### Main Page ##################################################


# 선택받은 옵션 key값
selected_options = []
for k, v in option_dict.items() :
    if v :
        selected_options.append(k)



''
'---'
''



col4, col5, col6 = st.columns([1, 3, 1])
with col4 :
    ''
with col5 :
    
    if st.session_state['carName'] == '그랜저' :
        st.subheader(mapper_grandeur_inv[st.session_state['name']])
        df = createNewDF(columns_carName=columns_grandeur)
        model = model_grandeur


    if st.session_state['carName'] == '투싼' :
        st.subheader(mapper_tucson_inv[st.session_state['name']])
        df = createNewDF(columns_carName=columns_tucson)
        model = model_tucson

    if st.session_state['carName'] == 'K5' :
        st.subheader(mapper_k5_inv[st.session_state['name']])
        df = createNewDF(columns_carName=columns_k5)
        model = model_k5


    if st.session_state['carName'] == '모닝' :
        st.subheader(mapper_morning_inv[st.session_state['name']])
        df = createNewDF(columns_carName=columns_morning)
        model = model_morning


    inputInfo(df, st.session_state['name'], st.session_state['year'], st.session_state['distance'], 
              st.session_state['fuel'], st.session_state['color'], selected_options)


    price = predict(model, df)
    margin = calcMargin(price)
    final_price = predictPrice(price, margin)
    

    margin = str(margin) + ' %'
    price = str(price) + ' 만원'
    final_price = str(final_price) + ' 만원'

    ''
    '---'
    ''

    st.write('마진율')
    st.error(margin)

    st.write('판매 예상 금액')
    st.success(final_price)
 

with col6 :
    ''


''
'---'
''








