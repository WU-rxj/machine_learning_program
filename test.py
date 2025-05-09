import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

data_folder = r'D:\titanic'
train_file = 'train.csv' 
test_file = 'test.csv' 

train_file_path = os.path.join(data_folder, train_file)
test_file_path = os.path.join(data_folder, test_file)

print(f"尝试读取训练文件: {train_file_path}")

if os.path.exists(train_file_path):
    df_train = pd.read_csv(train_file_path)
    print("\n训练数据成功加载。")
    print(df_train.head())
    print("\n原始数据基本信息:")
    df_train.info()

    # --- 2. 数据预处理 ---
    print("\n--- 开始数据预处理 ---")
    # 选择特征和目标变量
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    # 目标变量:
    target = 'Survived'

    # 创建一个副本进行操作，避免修改原始DataFrame
    df_processed = df_train[features + [target]].copy()

    # 处理缺失值
    # Age: 用中位数填充
    df_processed['Age'].fillna(df_processed['Age'].median(), inplace=True)
    # Embarked: 用众数填充
    df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0], inplace=True)
    # Fare: 用中位数填充 (train.csv中Fare通常没有缺失，但这是个好习惯)
    df_processed['Fare'].fillna(df_processed['Fare'].median(), inplace=True)

    # 转换类别特征
    # Sex: Label Encoding (male -> 0, female -> 1 or vice-versa)
    label_encoder_sex = LabelEncoder()
    df_processed['Sex'] = label_encoder_sex.fit_transform(df_processed['Sex'])

    # Embarked: One-Hot Encoding
    df_processed = pd.get_dummies(df_processed, columns=['Embarked'], prefix='Embarked', drop_first=True)
    
    # 从乘客姓名中提取称谓
    df_processed['Title'] = df_train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    
    # 将年龄和票价进行分箱处理
    df_processed['AgeBin'] = pd.cut(df_processed['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
    df_processed['FareBin'] = pd.qcut(df_processed['Fare'], q=4, labels=False)
    
    # 创建家庭大小特征
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    
    # 确保所有特征都经过适当的编码和标准化
    df_processed = pd.get_dummies(df_processed, columns=['Title', 'AgeBin', 'FareBin'], drop_first=True)

    print("\n预处理后数据预览 (前5行):")
    print(df_processed.head())
    print("\n预处理后数据基本信息:")
    df_processed.info()

    # 准备X (特征) 和 y (目标)
    X = df_processed.drop(target, axis=1)
    y = df_processed[target]
    
    # 确保所有列都是数值类型
    # 如果在get_dummies后仍有非数值列（理论上不应该），需要进一步检查
    # print(X.dtypes)


    # --- 3. 划分数据集 ---
    print("\n--- 划分训练集和测试集 ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # --- 4. 训练模型 ---
    print("\n--- 训练随机森林模型 ---")
    model = RandomForestClassifier(n_estimators=100, random_state=42) # 使用随机森林模型
    model.fit(X_train, y_train)
    print("模型训练完成。")

    # --- 5. 模型评估 ---
    print("\n--- 模型评估 ---")
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"训练集准确率: {train_accuracy:.4f}")

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"测试集准确率: {test_accuracy:.4f}")

    # --- 4.1 训练 XGBoost 模型 ---
    print("\n--- 训练 XGBoost 模型 ---")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    print("XGBoost 模型训练完成。")

    # --- 5.1 XGBoost 模型评估 ---
    print("\n--- XGBoost 模型评估 ---")
    y_pred_train_xgb = xgb_model.predict(X_train)
    train_accuracy_xgb = accuracy_score(y_train, y_pred_train_xgb)
    print(f"XGBoost 训练集准确率: {train_accuracy_xgb:.4f}")

    y_pred_test_xgb = xgb_model.predict(X_test)
    test_accuracy_xgb = accuracy_score(y_test, y_pred_test_xgb)
    print(f"XGBoost 测试集准确率: {test_accuracy_xgb:.4f}")

    # --- 6. 对测试集进行预测 ---
    print("\n--- 开始对测试集进行预测 ---")
    if os.path.exists(test_file_path):
        df_test_original = pd.read_csv(test_file_path)
        print("测试数据原始文件成功加载。")
        
        # 保存PassengerId用于最终提交文件
        passenger_ids = df_test_original['PassengerId']
        
        # 创建测试集副本进行预处理
        # 注意：这里的 features 列表是原始特征列表，不包含 'Survived'
        df_test_processed = df_test_original[features].copy()

        # 处理缺失值 - 使用训练集的统计数据
        # Age: 用训练集的中位数填充
        df_test_processed['Age'].fillna(df_train['Age'].median(), inplace=True)
        # Fare: 用训练集的中位数填充 (测试集中可能有缺失)
        df_test_processed['Fare'].fillna(df_train['Fare'].median(), inplace=True)
        # Embarked: 用训练集的众数填充
        df_test_processed['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)

        # 转换类别特征
        # Sex: 使用之前fit的LabelEncoder
        df_test_processed['Sex'] = label_encoder_sex.transform(df_test_processed['Sex'])

        # Embarked: One-Hot Encoding
        df_test_processed = pd.get_dummies(df_test_processed, columns=['Embarked'], prefix='Embarked', drop_first=True)
        
        # 对齐测试集的列与训练集X的列 (X_train.columns)
        # X_train.columns 已经包含了正确的独热编码后的列名
        # 使用 reindex 来确保测试集具有与训练模型时完全相同的特征列，包括顺序和缺失列（用0填充）
        df_test_processed = df_test_processed.reindex(columns=X_train.columns, fill_value=0)

        print("\n测试集预处理后数据预览 (前5行):")
        print(df_test_processed.head())
        print("\n测试集预处理后数据基本信息:")
        df_test_processed.info()

        # 进行预测 (随机森林)
        test_predictions_rf = model.predict(df_test_processed)

        # 进行预测 (XGBoost)
        test_predictions_xgb = xgb_model.predict(df_test_processed)

        # --- 7. 生成提交文件 ---
        print("\n--- 生成提交文件 ---")
        submission_df_rf = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_predictions_rf})
        submission_df_xgb = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_predictions_xgb})
        
        submission_file_path_rf = os.path.join(data_folder, 'submission_rf.csv')
        submission_df_rf.to_csv(submission_file_path_rf, index=False)
        print(f"随机森林提交文件已保存到: {submission_file_path_rf}")

        submission_file_path_xgb = os.path.join(data_folder, 'submission_xgb.csv')
        submission_df_xgb.to_csv(submission_file_path_xgb, index=False)
        print(f"XGBoost 提交文件已保存到: {submission_file_path_xgb}")

    else:
        print(f"错误: 测试文件 '{test_file_path}' 未找到。无法进行预测。")

else:
    print(f"错误: 训练文件 '{train_file_path}' 未找到。")
    print("请检查以下几点：")
    print(f"  - 文件夹路径 '{data_folder}' 是否正确？")
    print(f"  - 文件名 '{train_file}' 是否正确，并且该文件确实存在于上述文件夹中？")


