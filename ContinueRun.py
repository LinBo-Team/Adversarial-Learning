import os

# 不同参数对模型性能影响
print(11111)
os.system('python AdversarialModel.py --lambda_grl 0.8')
print(2222)
os.system('python AdversarialModel.py --lambda_grl 0.6')
print(33333)
os.system('python AdversarialModel.py --lambda_grl 0.4')
print(44444)
os.system('python AdversarialModel.py --lambda_grl 0.2')
print(55555)
os.system('python AdversarialModel.py --lambda_grl 0.1')

# ERM算法下的结果
# os.system('python main.py --source .\GearData\Condition123_4\Con123.mat --target .\GearData\Condition123_4\Con4.mat' )
# os.system('python main.py --source .\GearData\Condition234_1\Con234.mat --target .\GearData\Condition234_1\Con1.mat')
# os.system('python main.py --source .\GearData\Condition341_2\Con341.mat --target .\GearData\Condition341_2\Con2.mat')
# os.system('python main.py --source .\GearData\Condition412_3\Con412.mat --target .\GearData\Condition412_3\Con3.mat')

# CNN结果
# os.system('python ./network/CNN.py --source .\GearData\Condition123_4\Con123.mat --target .\GearData\Condition123_4\Con4.mat')
# os.system('python ./network/CNN.py --source .\GearData\Condition234_1\Con234.mat --target .\GearData\Condition234_1\Con1.mat')
# os.system('python ./network/CNN.py --source .\GearData\Condition341_2\Con341.mat --target .\GearData\Condition341_2\Con2.mat')
# os.system('python ./network/CNN.py --source .\GearData\Condition412_3\Con412.mat --target .\GearData\Condition412_3\Con3.mat')



