import requests
import json
import pandas as pd
import re

def Bus_inf(city,line):
    global bus_num  #全局变量，用于计算公交数目
    try:
        #获取数据
        url = 'https://restapi.amap.com/v3/bus/linename?s=rsv3&extensions=all&key=a5b7479db5b24fd68cedcf24f482c156&output=json&city={}&offset=1&keywords={}&platform=JS'.format(city,line)
        r = requests.get(url).text
        rt = json.loads(r)
        #读取当前公交线路主要信息
        dt = {}
        dt['line_name'] = rt['buslines'][0]['name'] #公交线路名字
        dt['polyname'] = rt['buslines'][0]['polyline'] #获取行驶路径
        bus_num+=1 #有效公交数+1
        
        """整理行车路径格式符合高德地图绘图工具的要求"""        
        b=re.split("[;]",dt['polyname'])
        res=""
        for i in range(len(b)):
            tmp=re.split("[,]",b[i])
            if len(res)==0:
                res=res+"["+tmp[0]+","+tmp[1]+"]"
            else:
                res=res+",["+tmp[0]+","+tmp[1]+"]"
        dt['polyname'] =res
        
        return pd.DataFrame(dt,index=[bus_num]) #下标index为“第几条公交线”
    except:
        return pd.DataFrame()  #读取数据失败，返空

if __name__=="__main__":
    bus_num=0  
    city='无锡' #需要查询公交信息的城市
    for_num=1000 
    all_buslines=pd.DataFrame()     
    for i in range(1,for_num+1):
        all_buslines=pd.concat([all_buslines,Bus_inf(city,str(i)+'路')])  #不加这个'路'可能优先获取地铁
    
    print("Bus_info函数遍历{}前{}路公交,有效公交线路数为：{}个的情况下：".format(city,for_num,bus_num))
    all_buslines.to_csv("{}前{}路公交(有效线路数：{})基本信息.csv".format(city,for_num,bus_num),index=False,encoding='utf-8-sig')

