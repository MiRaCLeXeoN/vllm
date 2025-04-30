import os
import time
import sys
from time import sleep

from vllm import LLM, SamplingParams
from utils import get_prompts
from vllm.utils import get_open_port


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--sd_method", type=str, default=None)
    parser.add_argument("--drafter_model", type=str, default=None)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--num_speculative_tokens", type=int, default=5)
    parser.add_argument("--backend", type=str, default="mp")
    return parser.parse_args()


def main(model, 
         dp_size, 
         tp_size,
         pp_size,
         drafter_model,
         sd_method,
         backend,
         local_dp_rank, 
         global_dp_rank, 
         dp_master_ip,
         dp_master_port):
    # os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    # os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    # os.environ["VLLM_DP_SIZE"] = str(dp_size)
    # os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    # os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    THROUGHPUT_PROMPTS = 2000

    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    device_ids = cuda_visible_devices.split(",")
    world_size = tp_size * dp_size
    world_size_with_dp = dp_size * world_size
    assert(world_size_with_dp <= len(device_ids))
    visible_device_ids = device_ids[local_dp_rank * world_size:(local_dp_rank + 1) * world_size]
    visible_devices = ",".join(visible_device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    print(f"[DP_Rank {local_dp_rank}] Using devices {visible_devices}")
    # Create an LLM.
    llm = LLM(
        model=model,
        max_model_len=500,
        data_parallel_size=1,
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        device="cuda",
        download_dir="/scr/dataset/yuke/zepeng/models",
        speculative_config={
            "model" : drafter_model,
            "method": sd_method,
            "num_speculative_tokens": 5,
        } if sd_method else None,
        gpu_memory_utilization=0.8,
        distributed_executor_backend=backend # NOTE(zt): mp-pp will replace mp in the __post_init__ ParallelConfig
    )

    warmup_prompts = get_prompts(10, 50)
    llm.generate(warmup_prompts)

    num_prompts = THROUGHPUT_PROMPTS // dp_size
    throughput_prompts = get_prompts(THROUGHPUT_PROMPTS, 50)
    throughput_prompts = throughput_prompts[local_dp_rank*num_prompts:(local_dp_rank+1)*num_prompts]
    print(f"[DP_Rank {local_dp_rank}][Throughput-Test]Number of Prompts: {len(throughput_prompts)}")

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    start_time = time.time()
    outputs = llm.generate(throughput_prompts, sampling_params)
    end_time = time.time()
    print(f"[DP_Rank {local_dp_rank}][Throughput-Test]Time taken: {end_time - start_time} seconds")
    print(f"[DP_Rank {local_dp_rank}][Throughput-Test]Number of outputs: {len(outputs)}")

    latency_prompts = get_prompts(4, 50)

    start_time = time.time()
    outputs = llm.generate(latency_prompts, sampling_params)
    end_time = time.time()
    print(f"[DP_Rank {local_dp_rank}][Latency-Test]Time taken: {end_time - start_time} seconds")
    print(f"[DP_Rank {local_dp_rank}][Latency-Test]Number of outputs: {len(outputs)}")

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":

    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    pp_size = args.pp_size

    if dp_size > 1 and args.version != 1:
        raise RuntimeError("Not supported")
    
    if args.version == 1:
        os.environ["VLLM_USE_V1"] = "1"
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

    dp_master_ip = "127.0.0.1"
    dp_master_port = get_open_port()

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(range(0, dp_size)):
        proc = Process(target=main,
                        kwargs={
                            "model": args.model,
                            "dp_size": dp_size,
                            "tp_size": tp_size,
                            "pp_size": pp_size,
                            "drafter_model": args.drafter_model,
                            "sd_method": args.sd_method,
                            "backend": args.backend,
                            "local_dp_rank": local_dp_rank,
                            "global_dp_rank": global_dp_rank,
                            "dp_master_ip": dp_master_ip,
                            "dp_master_port": dp_master_port,
                        })
        procs.append(proc)
    for proc in procs:
        proc.start()

    exit_code = 0
    for proc in procs:
        proc.join(timeout=600)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that "
                  f"didn't stop within 10 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
