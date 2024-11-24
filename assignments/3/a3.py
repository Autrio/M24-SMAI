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
        "assignments.3.2.1",
        "assignments.3.2.2",
        "assignments.3.2.3",
        "assignments.3.2.4",
        "assignments.3.2.5",
        "assignments.3.2.6",
        "assignments.3.3.1",
        "assignments.3.3.2",
        "assignments.3.3.3",
        "assignments.3.3.4",
        "assignments.3.3.5",
        "assignments.3.4.2",
        "assignments.3.4.3",
        "assignments.3.4.4"
    ]
    
    for command in commands:    
        run(command)

if __name__ == "__main__":
    main()
