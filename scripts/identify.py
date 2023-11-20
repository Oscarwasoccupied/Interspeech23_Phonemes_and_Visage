import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import seaborn as sns

def t_test(data, type):
    data = np.array(data)
    t_value = [6.314, 2.92, 2.353, 2.132, 2.015,1.943, 1.895, 1.860, 1.833, 1.812]
    N = len(data)
    CI_u = data.mean() + t_value[N-2]*data.std()/np.sqrt(N)
    # print(type+" 1-CI_u: ", 1 - CI_u)
    return 1 - CI_u

def main(num_experiments=10):
    # print(pd.__version__)
    path = './estimations/'
    file_list = os.listdir(path)
    print(file_list)
    df = pd.DataFrame()
    for file in file_list:
        mean = np.loadtxt(path+file+'/'+file+'.txt', delimiter=',')[:,2]
    
        result = []
        for i in range(num_experiments):
            E = pd.read_csv(path+file+'/' + file +'_times'+str(i)+'.csv')

            # print(file +'_times'+str(i))

            m = mean[i] 
            epsilon = mean_squared_error(E[' prediction'], E[' label'])
            # print("MSE: ", epsilon)

            C = E.copy()
            C[' prediction'] = m    
            epsilon_C = mean_squared_error(C[' prediction'], C[' label'])
            # print("MSE: ", epsilon_C)
            result.append(epsilon/epsilon_C)

        t_100 = t_test(result, "100%")
        t_75 = t_test(sorted(result)[:int(0.75 * len(result))], "75%")
        t_50 = t_test(sorted(result)[:int(0.5 * len(result))], "50%")
        s = pd.Series({'pair': file, '100%': t_100, '75%': t_75, '50%': t_50})
        s = pd.DataFrame(s).transpose()
        df = pd.concat([df, s], ignore_index=True)
        # print("s: ", s)
        # print("type(s): ", type(s)) 
        # print("head of s: ", s.head())
        # print("head of df: ", df.head())
        # print("type(df): ", type(df))
        
    df.to_csv("t_test_result.csv", index=False)
    sns.set_palette("YlOrBr", 3)
    df.sort_values(by=['50%'], ascending=False).plot(x="pair", y=["100%", "75%", "50%"], kind="bar",figsize=(9,6),title="Result of t-test") 
    plt.title("Result of t-test", fontsize=16,fontweight='bold')
    plt.xlabel("phoneme - AM pair", fontsize=15,fontweight='bold')
    plt.ylabel("1-CIu",fontsize=15,fontweight='bold')
    plt.legend(["100% val set","75% val set","50% val set"])
    plt.yticks(size=15)
    plt.xticks(size=12, rotation=45)
    # plt.rcParams['font.sans-serif'] = ['KaiTi'] 
    plt.rcParams['axes.unicode_minus'] = False  
    plt.tight_layout()
    plt.savefig("t_test_result.png")

if __name__ == "__main__":
    main()