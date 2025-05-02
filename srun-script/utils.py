from typing import List, Dict
import os


def get_prompts(prompt_size: int, string_length: int, prompt_folder_path: str = "./prompts") -> List[str]:
    prompt_file_path_list = os.listdir(prompt_folder_path)
    prompt_file_path_list = [os.path.join(prompt_folder_path, file_path) for file_path in prompt_file_path_list]

    total_string_size = prompt_size * string_length
    prompts = ""
    while len(prompts) < total_string_size:
        for prompt_file_path in prompt_file_path_list:
            with open(prompt_file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if len(line) > 0:
                        prompts += line + " "

    words = prompts.split()
    
    result = []
    for i in range(0, len(words), string_length):
        chunk = " ".join(words[i:i+string_length])
        result.append(chunk)
        if len(result) >= prompt_size:
            break
    
    return result

def get_average_profile(log_path: str) -> Dict[str, float]:
    profile = {}
    dp = set()
    with open(log_path, "r") as f:
        lines = f.readlines()
        # [DP_Rank 1][Throughput-Test]Number of outputs: 1000
        # [DP_Rank 1][Latency-Test]Time taken: 0.18574857711791992 seconds
        for line in lines:
            if "[Throughput-Test]Time taken:" in line or\
                "[Latency-Test]Time taken:" in line:
                if "[Throughput-Test]" in line:
                    key = "Throughput-Test"
                elif "[Latency-Test]" in line:
                    key = "Latency-Test"
                else:
                    raise ValueError(f"Invalid line: {line}")
                time_taken = float(line.split(":")[1].strip().split(" ")[0])
                if key not in profile:
                    profile[key] = [time_taken, 1]
                else:
                    profile[key][0] += time_taken
                    profile[key][1] += 1
            if "DP_Rank" in line:
                dp.add(line.split(" ")[1])
    throughput_total_time = 0
    throughput_total_count = 0
    latency_total_time = 0
    latency_total_count = 0
    print(profile)
    for key in profile:
        if "Throughput-Test" in key:
            throughput_total_time += profile[key][0]
            throughput_total_count += profile[key][1]
        elif "Latency-Test" in key:
            latency_total_time += profile[key][0]
            latency_total_count += profile[key][1]
    output_throughput = throughput_total_time / throughput_total_count
    output_latency = latency_total_time / latency_total_count
    return output_throughput, output_latency
        
def download_from_huggingface(model_name: str, output_dir: str):
    from huggingface_hub import snapshot_download
    local_path = snapshot_download(
        repo_id=model_name,
        local_dir=output_dir,
        local_dir_use_symlinks=True
    )
    print(f"Model downloaded to: {local_path}")
    
    
    
if __name__ == "__main__":
    # prompts = get_prompts(4, 50)
    # print(len(prompts))
    # for prompt in prompts:
    #     print(len(prompt))
    
    # file_path = "./mp_v1_sd_3D/opt_sd_eagle_dp_2_tp_4_pp_1.log"
    # throughput, latency = get_average_profile(file_path)
    # print(f"Throughput: {throughput}, Latency: {latency}")
    
    # import json
    # path_list = os.listdir("./mp_v1_3D")
    # json_data = {}
    # for path in path_list:
    #     if path == "mp_v1_3D.json":
    #         continue
    #     print(path)
    #     throughput, latency = get_average_profile(os.path.join("./mp_v1_3D", path))
    #     json_data[path] = {
    #         "throughput": throughput,
    #         "latency": latency
    #     }
    # with open("./mp_v1_3D/mp_v1_3D.json", "w") as f:
    #     json.dump(json_data, f)

    download_from_huggingface(
        "facebook/opt-125m",
        "/scr/dataset/yuke/zepeng/models/models--facebook--opt-125m"
    )