# Tkinterライブラリのインポート
import os
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
from bai_sim import bai_sim
sns.set()

ask_file = 'USDJPY_Candlestick_1_D_ASK_01.07.2003-31.07.2021.csv'
bid_file = 'USDJPY_Candlestick_1_D_BID_01.07.2003-31.07.2021.csv'

def validation(before_word, after_word):
    return ((after_word.isdecimal()) and (len(after_word) <= 4)) or (len(after_word) == 0)


class Application(tk.Frame):
    # self=tk.Frame

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        sv = tk.StringVar()

        self.start = tk.Button(self.master, text="開始", fg="blue", command=self.start)
        self.start.place(x=280, y=70)

        self.exit = tk.Button(self.master, text="終了", fg="red", command=self.master.destroy)
        self.exit.place(x=340, y=70)

        self.Number = tk.Entry(width=20, textvariable=sv)
        self.Number.place(x=90, y=70)

        v_cmd = (self.Number.register(validation), '%s', '%P')

        self.Number.configure(validate='key', vcmd=v_cmd)

        self.lbl_n1 = tk.Label(text='初期投資額')
        self.lbl_n1.place(x=20, y=70)

        self.lbl_n2 = tk.Label(text='(万円)')
        self.lbl_n2.place(x=210, y=70)

    def start(self):
        x = int(self.Number.get()) * 10000
        i = bai_sim(x, ask_file, bid_file)
        User(tk.Toplevel(), i)


class User(tk.Frame):
    def __init__(self, master, result):
        super().__init__(master)
        self.pack()
        master.geometry("600x300")
        master.title("結果")
        self.result = result

        result1 = round(self.result[0])
        result2 = round(self.result[1])
        result3 = round(self.result[2])
        result4 = self.result[3]
        result5 = self.result[4]
        result6 = self.result[5]
        result7 = self.result[6]

        self.label1 = tk.Label(master, text='・現在の金額: ' + "{:,}".format(result1) + ' (円)')
        self.label1.place(x=10, y=40)

        self.label2 = tk.Label(master, text='・差額: ' + "{:,}".format(result2) + ' (円)')
        self.label2.place(x=10, y=80)

        self.label3 = tk.Label(master, text='・勝率: ' + str(result3) + ' (%)')
        self.label3.place(x=10, y=120)

        self.label4 = tk.Label(master, text='・取引回数: ' + str(result4) + ' (回)')
        self.label4.place(x=310, y=40)

        self.label5 = tk.Label(master, text='・勝利回数: ' + str(result5) + ' (回)')
        self.label5.place(x=310, y=80)

        self.label6 = tk.Label(master, text='・敗北回数: ' + str(result6) + ' (回)')
        self.label6.place(x=310, y=120)

        figA = plt.figure()
        axA = figA.subplots(1, 2)
        figA.subplots_adjust(bottom=0.2)
        axA[0].plot(result7['sim'])
        axA[1].plot(result7['ans'])

        plt.show()

        try:
            os.remove('all.png')
            os.remove('result.csv')
        except OSError as err:
            pass
        figA.savefig("all.png")
        result7.to_csv("result.csv")



root = tk.Tk()
root.geometry("500x200")
app = Application(master=root)
app.mainloop()
