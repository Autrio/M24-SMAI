import subprocess

def run(func, query_type):
    command = f"python3 -m {func} -q {query_type}"
    print(f"Running command: {command}\n")
    
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for line in iter(process.stderr.readline, ''):
            print(f"{line.strip()}")
        for line in iter(process.stdout.readline, ''):
            print(f"Output: {line.strip()}")
        
        process.stdout.close()
        process.stderr.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"Command failed with return code {return_code}")

def main():
    for i in [1,2,3,4,5]:
        run("assignments.1.a1-knn",i)
    for i in [1,2,3,4]:
        run("assignments.1.a1-lr",i)

if __name__ == "__main__":
    main()
