import subprocess


# 要运行的Python程序路径
program_path = "main.py"

# 不同的参数组合,表示不同的
# parameters_of_lamuda =['200','300','400','500','600','700','800']
parameters_of_method = ['1','2','3']
len1=len(parameters_of_method)
parameters_of_division=['1','2']
len2=len(parameters_of_division)

# parameters_of_clients=['50']#['20','30','40','50','60','70','80']
# len2=len(parameters_of_clients)
parameters_of_stackelberg=['1']#['0','1']
for i,type in enumerate(parameters_of_stackelberg):
    for j, para in enumerate(parameters_of_division):
        for k,args in enumerate(parameters_of_method):
            print(f"正在运行第 {i*len1*len2 + j*len1+k+1} 次")
            # cmd = ["python", program_path ,"--num_of_clients",args,"--is_stackelberg",type,"--lamuda",para]
            cmd = ["python", program_path, "--method", args, "--is_stackelberg", type, "--method_division", para]
            result = subprocess.run(cmd, encoding="utf-8", capture_output=True, text=True)
            print(result.stdout)  # This will print the program's output
            # print("输出:", result.stdout)
            print(f"第 {i*len1*len2 + j*len1+k+1} 次运行完成")
# result1 = subprocess.run(cmd1, encoding="utf-8", capture_output=True, text=True)
# print(result1.stdout)
# result2 = subprocess.run(cmd2, encoding="utf-8", capture_output=True, text=True)
# print(result2.stdout)
# result3 = subprocess.run(cmd3, encoding="utf-8", capture_output=True, text=True)
# print(result3.stdout)
# result4 = subprocess.run(cmd4, encoding="utf-8", capture_output=True, text=True)
# print(result4.stdout)