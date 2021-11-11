# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


def keras(file):
    '''
    1. データの準備
    '''
    data_type = {'time': str, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
    name_list = ['time', 'open', 'high', 'low', 'close', 'volume']
    file_name = file
    df = pd.read_csv(filepath_or_buffer=file_name, dtype=data_type, header=0, names=name_list, parse_dates=['time'])
    df_t = df.set_index('time')
    pd.set_option('display.max_rows', len(df['close']))

    # T日移動平均
    T = 75  # 移動平均
    df_t['mov_Av'] = df_t['close'].rolling(T).mean()
    mov_Av = np.array(df_t['mov_Av'])

    # 乖離率
    df_t['mov_dev'] = np.nan
    df_t['mov_dev'] = (df_t['close'] - df_t['mov_Av']) / df_t['mov_Av'] * 100
    mov_dev = np.array(df_t['mov_dev'])

    # SP波動法
    sp = 1.5  # SP
    day = []
    owarine = []
    data, data2 = [], []
    ten, tei = 0, 0
    result = []
    inc = []
    close = np.array(df['close'])

    for i in range(len(df['close'])):
        day.append(df['time'][i])
        owarine.append(df['close'][i])

    for i in range(len(day)):
        set1 = []
        set1.append(day[i])
        set1.append(owarine[i])
        data2.append(set1)

    def bottom(x):
        y = ten * (200 - sp) / (200 + sp)
        if (owarine[x] < y):
            return 1
        return 0

    def top(x):
        y = tei * (200 + sp) / (200 - sp)
        if (owarine[x] > y):
            return 1
        return 0

    saidai = np.argmax(owarine)
    result.append(data2[saidai])
    max_s = saidai
    ten = owarine[saidai]
    mode = 0

    for i in range(saidai, -1, -1):

        if mode == 0:
            if (bottom(i) == 1):
                tei = owarine[i]
                max_s = owarine.index(owarine[i])
                result.append([data2[i], 0])
                mode = 1
            elif (owarine[i] >= ten):
                ten = owarine[i]
                max_s = owarine.index(owarine[i])
                result.pop()
                result.append([data2[i], 1])

        if mode == 1:
            if (top(i) == 1):
                ten = owarine[i]
                max_s = owarine.index(owarine[i])
                result.append([data2[i], 1])
                mode = 0
            elif (owarine[i] <= tei):
                tei = owarine[i]
                max_s = owarine.index(owarine[i])
                result.pop()
                result.append([data2[i], 0])

    result.pop()
    result.reverse()
    max_s = saidai
    ten = owarine[saidai]
    mode = 0

    for i in range(saidai, len(owarine), 1):

        if mode == 0:
            if (bottom(i) == 1):
                tei = owarine[i]
                max_s = owarine.index(owarine[i])
                result.append([data2[i], 0])
                mode = 1
            elif (owarine[i] > ten):
                ten = owarine[i]
                max_s = owarine.index(owarine[i])
                result.pop()
                result.append([data2[i], 1])

        if mode == 1:
            if (top(i) == 1):
                ten = owarine[i]
                max_s = owarine.index(owarine[i])
                result.append([data2[i], 1])
                mode = 0
            elif (owarine[i] < tei):
                tei = owarine[i]
                max_s = owarine.index(owarine[i])
                result.pop()
                result.append([data2[i], 0])

    result.pop()

    # 上昇度
    start = 0
    flag_1 = True
    cnd_l = []

    for i in range(0, len(result)):
        cnd_l.append(df_t.index.get_loc(result[i][0][0]))

    if result[0][1] == 0:
        for i in cnd_l:
            if flag_1:
                list_0 = [0] * (i + 1 - start)
                inc.extend(list_0)
                start = i + 1
                flag_1 = False
            else:
                list_1 = [1] * (i + 1 - start)
                inc.extend(list_1)
                start = i + 1
                flag_1 = True
    else:
        for i in cnd_l:
            if flag_1:
                list_1 = [1] * (i + 1 - start)
                inc.extend(list_1)
                start = i + 1
                flag_1 = False
            else:
                list_1 = [0] * (i + 1 - start)
                inc.extend(list_1)
                start = i + 1
                flag_1 = True

    # 上昇度の末尾確認
    if not len(close) == len(inc):
        if inc[-1] == 0:
            if close[len(inc)] < close[len(inc) + 1]:
                list_1 = [1] * (len(close) - len(inc))
                inc.extend(list_1)
            else:
                list_0 = [0] * (len(close) - len(inc))
                inc.extend(list_0)
        else:
            if close[len(inc)] < close[len(inc) + 1]:
                list_1 = [1] * (len(close) - len(inc))
                inc.extend(list_1)
            else:
                list_0 = [0] * (len(close) - len(inc))
                inc.extend(list_0)

    # Pandas+上昇度
    df['inc'] = inc
    df_t['inc'] = inc
    inc = np.array(df_t['inc'])

    # モデル準備
    t = T
    cnt_a = len(df) // t
    amr = len(df) - (t * cnt_a)
    mov_dev = mov_dev[T + amr::]
    inc = inc[T + amr::]
    data = np.array(mov_dev).reshape(-1, t, 1)
    target = np.array(inc).reshape(-1, t)
    d_train, d_val, t_train, t_val = train_test_split(data, target, test_size=0.5, shuffle=False)  # 予測年比率

    '''                                                                                                                                                                                                                                                                                 
    2. モデルの構築                                                                                                                                                                                                                                                                           
    '''
    model = Sequential()
    model.add(LSTM(75, activation='tanh', recurrent_activation='sigmoid',
                   kernel_initializer='glorot_normal', recurrent_initializer='orthogonal'))
    model.add(Dense(1, activation='linear'))

    '''                                                                                                                                                                                                                                                                                 
    3. モデルの学習                                                                                                                                                                                                                                                                           
    '''
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=optimizer, metrics=['accuracy'], loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    yen_history = model.fit(d_train, t_train, epochs=200, batch_size=12, verbose=2, validation_data=(d_val, t_val),
                            callbacks=[es])

    '''                                                                                                                                                                                                                                                                                 
    4. 予測形成                                                                                                                                                                                                                                                                      
    '''
    d_l = d_val.reshape(-1, 1, 1)
    pred = model.predict(d_l)
    t_l = t_val.reshape(-1, )
    p_l = pred.reshape(-1, )
    p_l = np.where((p_l > 0.07) & (p_l < max(p_l)), 1, 0)

    '''                                                                                                                                                                                                                                                                                 
    5. 形成データ整理                                                                                                                                                                                                                                                                           
    '''
    df_d = df_t.tail(len(t_l)).copy()
    df_d['inc_p'] = p_l

    res_s = 0
    res_l = []
    cres_l = []

    for i, g in df_d.groupby([(df_d.inc_p.shift() != df_d.inc_p).cumsum()]):
        res_s += len(g.inc_p.tolist()) - 1
        if len(g.inc_p.tolist()) >= 2:
            if sum(g.inc_p.tolist()) > 0:
                if 'bid' in file.lower():
                    res_l.append(res_s)
            elif sum(g.inc_p.tolist()) == 0:
                if 'ask' in file.lower():
                    res_l.append(res_s)

    for i in res_l:
        cres_l.append(df_d.iloc[i, 3])

    return cres_l, res_l, df_d


'''
6. 売買シミュレーション
'''


def bid_process(moto, ask_df, bid_df, ask_inc, bid_inc):
    spred = 0.005  # スプレッド
    lev = 25  # レバレッジ
    per = 0.1  # 投資割合
    r = 0.9  # α
    cut = 2  # 損切り値
    q = 0.05  # 利食い開始値
    result = moto  # 資金
    ask_lot = 0  # 買い玉
    bid_lot = 0  # 売り玉
    ask_max = 0
    bid_min = 0
    res_prol = []
    ask_close = ask_df['close']
    ask_open = ask_df['open']
    bid_close = bid_df['close']
    bid_open = bid_df['open']
    ask_f = False
    bid_f = False
    win = 0
    lose = 0
    cnt = 0
    for i in range(len(ask_df) - 1):
        if result >= 0:
            if ask_inc[i] == 0 and ask_inc[i + 1] == 1:
                ask_position = (ask_close[i]) + spred
                ask_p = ask_position
                ask_max = ask_p
                ask_f = True
                cnt += 1
                ask_lot += (int((result * lev * per) / ask_position))
            if ask_lot <= 0 and ask_inc[i] == 1 and ask_inc[i + 1] == 1:
                ask_position = (ask_close[i]) + spred
                ask_p = ask_position
                ask_max = ask_p
                ask_f = True
                cnt += 1
                ask_lot += (int((result * lev * per) / ask_position))

            if bid_inc[i] == 1 and bid_inc[i + 1] == 0:
                bid_position = (bid_close[i]) - spred
                bid_p = bid_position
                bid_min = bid_p
                bid_f = True
                cnt += 1
                bid_lot += (int((result * lev * per) / bid_position))
            if bid_lot <= 0 and bid_inc[i] == 0 and bid_inc[i + 1] == 0:
                bid_position = (bid_close[i]) - spred
                bid_p = bid_position
                bid_min = bid_p
                bid_f = True
                cnt += 1
                bid_lot += (int((result * lev * per) / bid_position))

        if ask_f or bid_f:
            if ask_f:
                if int(bid_inc[i]) == 1:
                    if ask_p < bid_close[i]:
                        ask_max = bid_close[i]
                    if ask_p - cut <= bid_close[i]:
                        if ask_p - cut >= bid_open[i + 1]:
                            tmp = (bid_open[i + 1] - ask_p - spred) * ask_lot
                            result += tmp
                            lose += 1
                            cnt += 1
                            ask_f = False
                            ask_lot = 0

                    if ask_f:
                        if bid_close[i] >= ask_p + q:
                            if bid_close[i] >= ask_p + (ask_max - ask_p) * r:
                                tmp = ((ask_max - ask_p) * r - spred) * ask_lot
                                result += tmp
                                win += 1
                                cnt += 1
                                ask_f = False
                                ask_lot = 0

                    if ask_f:
                        if ask_p + (ask_max - ask_p) * r <= bid_close[i]:
                            if ask_p + (ask_max - ask_p) * r >= bid_open[i + 1]:
                                tmp = (bid_open[i + 1] - ask_p - spred) * ask_lot
                                result += tmp
                                if tmp < 0:
                                    lose += 1
                                    cnt += 1
                                    ask_f = False
                                    ask_lot = 0
                                else:
                                    win += 1
                                    cnt += 1
                                    ask_f = False
                                    ask_lot = 0

            if bid_f:
                if int(ask_inc[i]) == 0:
                    if bid_p > ask_close[i]:
                        bid_min = ask_close[i]
                    if bid_p + cut >= ask_close[i]:
                        if bid_p + cut <= ask_open[i + 1]:
                            tmp = (bid_p - ask_open[i + 1] - spred) * bid_lot
                            result += tmp
                            lose += 1
                            cnt += 1
                            bid_f = False
                            bid_lot = 0

                    if bid_f:
                        if ask_close[i] <= bid_p - q:
                            if ask_close[i] <= bid_p - (bid_p - bid_min) * r:
                                tmp = ((bid_p - bid_min) * r - spred) * bid_lot
                                result += tmp
                                win += 1
                                cnt += 1
                                bid_f = False
                                bid_lot = 0

                    if bid_f:
                        if ask_close[i] >= bid_p - q:
                            if bid_p - (bid_p - bid_min) * r >= ask_close[i]:
                                if bid_p - (bid_p - bid_min) * r <= ask_open[i + 1]:
                                    tmp = (bid_p - ask_open[i + 1] - spred) * bid_lot
                                    result += tmp
                                    if tmp < 0:
                                        lose += 1
                                        cnt += 1
                                        bid_f = False
                                        bid_lot = 0
                                    else:
                                        win += 1
                                        cnt += 1
                                        bid_f = False
                                        bid_lot = 0
        res_prol.append(result)
    res_prol.append(result)

    return res_prol, cnt, win, lose, result


'''
7. 売買シミュレーション呼び出し・結果
'''


def bai_sim(user_in, ask_file, bid_file):
    moto = user_in

    ask_res = keras(ask_file)
    bid_res = keras(bid_file)

    ask_df = ask_res[2]
    bid_df = bid_res[2]

    prcss_res = bid_process(moto, ask_df, bid_df, ask_df['inc_p'], bid_df['inc_p'])
    prcss_ans = bid_process(moto, ask_df, bid_df, ask_df['inc'], bid_df['inc'])

    moto_l = prcss_res[0]
    res_cnt = prcss_res[1]
    win_cnt = prcss_res[2]
    lose_cnt = prcss_res[3]
    moto_res = prcss_res[4]

    mgn = (moto_res / moto)
    win = moto_res - moto
    win_r = mgn * 100

    moto_al = prcss_ans[0]
    moto_ares = prcss_ans[4]

    res_diff = abs(moto_ares - moto_res)
    bid_df['sim'] = moto_l
    bid_df['ans'] = moto_al

    return moto_res, win, win_r, res_cnt, win_cnt, lose_cnt, bid_df, res_diff
