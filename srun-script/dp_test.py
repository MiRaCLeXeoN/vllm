import subprocess

total = 8

combinations = []

for a in range(1, total+1):
    for b in range(1, total+1):
        for c in range(1, total+1):
            if a*b*c == total:
                combinations.append((a, b, c))

print(combinations)

comb = (2, 2, 2)

command = [
    "python",
    "./data_parallel.py",
    "--version", "1",
    "--model", "/scr/dataset/yuke/zepeng/models/models--facebook--opt-6.7b/snapshots/a45aa65bbeb77c1558bc99bedc6779195462dab0",
    "--sd_method", "eagle",
    "--drafter_model", "/scr/dataset/yuke/zepeng/models/models--yuhuili--EAGLE-LLaMA3-Instruct-8B/snapshots/02d97adb0cacd1a50d4af8db61f4aca424ba2531",
    "--dp_size", f"{comb[0]}",
    "--pp_size", f"{comb[1]}",
    "--tp_size", f"{comb[2]}",
    "--backend", "mp",
]

subprocess.run(command)
                