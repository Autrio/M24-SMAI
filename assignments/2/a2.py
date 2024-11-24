import subprocess

def run(func):
    command = f"python3 -m {func}"
    print(f"Running command: {command}\n")
    
    with subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for line in iter(process.stderr.readline, ''):
            print(f"Error: {line.strip()}")
        for line in iter(process.stdout.readline, ''):
            print(f"Output: {line.strip()}")
        
        process.stdout.close()
        process.stderr.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"Command failed with return code {return_code}")

def main():
    commands = [
        "assignments.2.task3.1",
        "assignments.2.task3.2",
        "assignments.2.task4.1",
        "assignments.2.task4.2",
        "assignments.2.task4.3",
        "assignments.2.task5.2",
        "assignments.2.task6.1",
        "assignments.2.task6.2",
        "assignments.2.task6.3",
        "assignments.2.task6.4",
        "assignments.2.task8.1",
        "assignments.2.task9.1",
        "assignments.2.task9.2"
    ]
    
    for command in commands:
        run(command)

if __name__ == "__main__":
    main()
