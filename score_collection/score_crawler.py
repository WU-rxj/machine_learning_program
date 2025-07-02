import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
from datetime import datetime
from webdriver_manager.chrome import ChromeDriverManager

# 配置参数
EXCEL_PATH = r"d:\vscode_program\score_collection\数经班名单.xlsx"
URL = "https://table.nju.edu.cn/external-apps/fe886b54-8cec-4e7e-a5b6-3ed47d72f13c/?page_id=Z8IX"
OUTPUT_PATH = r"d:\vscode_program\score_collection\学位学分绩结果_{}.xlsx".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

# 检查Excel文件是否存在
if not os.path.exists(EXCEL_PATH):
    print(f"错误：找不到Excel文件 {EXCEL_PATH}")
    exit(1)

# 读取Excel文件
try:
    # 尝试使用pandas读取Excel文件
    df = pd.read_excel(EXCEL_PATH)
    
    # 检查是否有数据
    if df.empty:
        print("错误：Excel文件中没有数据")
        exit(1)
    
    # 打印列名，以便确认
    print("Excel文件列名:", df.columns.tolist())
    
    # 明确指定使用"学号"列和"姓名"列
    student_id_col = "学号"
    name_col = "姓名"
    
    # 检查指定的列是否存在
    if student_id_col not in df.columns:
        print(f"错误：Excel文件中没有'{student_id_col}'列")
        exit(1)
    if name_col not in df.columns:
        print(f"错误：Excel文件中没有'{name_col}'列")
        exit(1)
    
    print(f"将使用列 '{student_id_col}' 作为学号，'{name_col}' 作为姓名")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame()
    
except Exception as e:
    print(f"读取Excel文件时出错: {str(e)}")
    exit(1)

# 初始化WebDriver
try:
    # 设置Chrome选项
    options = webdriver.ChromeOptions()
    # 添加必要的选项以避免检测
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-infobars')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option('useAutomationExtension', False)
    
    # 初始化Chrome浏览器
    # 使用ChromeDriverManager自动管理ChromeDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    # 执行CDP命令以避免被检测为自动化测试
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })
    
    driver.maximize_window()
    driver.get(URL)
    print("成功打开网站")
    
    # 等待页面完全加载
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, "//input[@placeholder='学号']"))
    )
    
    # 创建结果列表
    results = []
    
    # 遍历每个学生信息
    for index, row in df.iterrows():
        student_id = str(row[student_id_col])
        name = str(row[name_col])
        
        print(f"处理学生: {name}, 学号: {student_id}")
        
        try:
            # 等待所有元素加载完成
            WebDriverWait(driver, 10).until(lambda d: len(d.find_elements(By.XPATH, "//input")) >= 3)
            
            # 填写学号
            student_id_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='学号']"))
            )
            student_id_input.clear()
            student_id_input.send_keys(student_id)
            print(f"成功填写学号: {student_id}")
            time.sleep(1) # 增加等待时间
            
            # 填写姓名
            name_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//input[@placeholder='姓名']"))
            )
            name_input.clear()
            name_input.send_keys(name)
            print(f"成功填写姓名: {name}")
            time.sleep(1) # 增加等待时间
            
            # 选择"不等于"
            select_element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "select"))
            )
            select = Select(select_element)
            select.select_by_visible_text("不等于")
            print("成功选择'不等于'")
            time.sleep(1) # 增加等待时间
            
            # 填写手机号旁边的字段为"1"
            try:
                # 尝试使用更精确的XPath定位：查找<select>元素后面的第一个<input>兄弟元素
                third_field = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//select/following-sibling::input[1]"))
                )
                third_field.clear()
                third_field.send_keys("1")
                print("成功填写手机号旁边的字段为'1' (精确XPath)")
                time.sleep(1) # 增加等待时间
            except Exception as e:
                print(f"尝试使用精确XPath定位手机号旁边的字段失败，尝试使用通用方法。错误: {str(e)}")
                # 回退到原始方法：通过查找所有文本输入框的索引来定位
                inputs = driver.find_elements(By.XPATH, "//input[@type='text']")
                if len(inputs) >= 3:
                    third_field = inputs[2]  # 第三个input元素
                    third_field.clear()
                    third_field.send_keys("1")
                    print("成功使用通用方法填写手机号旁边的字段为'1'")
                    time.sleep(1) # 增加等待时间
                else:
                    print("未找到足够的文本输入框来填写手机号旁边的字段。跳过当前学生。")
                    continue # 如果找不到目标元素，跳过当前学生
            
            # 点击查询按钮
            search_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), '查询')]"))
            )
            search_button.click()
            print("成功点击查询按钮")
            time.sleep(3)  # 增加等待时间，确保页面有足够时间加载
            
            # 等待结果表格加载并确保至少有一行数据
            WebDriverWait(driver, 20).until( # 增加等待超时时间
                EC.presence_of_element_located((By.XPATH, "//table//tbody//tr")) # 等待表格内部的行元素
            )
            print("成功等待结果表格及数据加载")
            time.sleep(2)  # 额外等待以确保数据完全加载完成
            
            # 获取结果表格
            table = driver.find_element(By.TAG_NAME, "table")
            
            # 获取表头
            headers = [th.text for th in table.find_elements(By.XPATH, ".//th")]
            
            # 获取所有行
            rows = table.find_elements(By.XPATH, ".//tbody/tr")
            
            # 处理每一行数据
            for row_element in rows:
                cells = row_element.find_elements(By.XPATH, ".//td")
                row_data = [cell.text for cell in cells]
                
                # 创建包含当前学生信息和结果的字典
                result_dict = {
                    "学号": student_id,
                    "姓名": name
                }
                
                # 添加表格中的其他数据
                for i, header in enumerate(headers):
                    if i < len(row_data):
                        result_dict[header] = row_data[i]
                
                results.append(result_dict)
            
            print(f"成功获取 {name} 的学位学分绩信息")
            
        except Exception as e:
            print(f"处理学生 {name} 时出错: {str(e)}")
            # 继续处理下一个学生
            continue
    
    # 将结果转换为DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # 保存结果到Excel文件
        results_df.to_excel(OUTPUT_PATH, index=False)
        print(f"成功将结果保存到: {OUTPUT_PATH}")
    else:
        print("没有获取到任何结果")
    
except Exception as e:
    print(f"浏览器操作时出错: {str(e)}")

finally:
    # 关闭浏览器
    if 'driver' in locals():
        driver.quit()
    print("程序执行完毕")