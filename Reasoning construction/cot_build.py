import argparse
import copy
import json
import os
import random
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from call_llm import *

from datasets import load_dataset
from tqdm import tqdm

from cot_prompt_en import verify_prompt, query_prompt_init_with_decompose, query_prompt_init, search_strategies, \
    gen_prompt_w_label, reformat_to_complex_cot_prompt, get_final_response_prompt

# 这里是你的梯子，如果在国外可以注释掉
os.environ["HTTP_PROXY"] = "http://127.0.0.1:4780"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:4780"


def extract_bracket_content(text):
    # Extract content between the first '{' and the last '}'
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None


def parse_gpt_response(response):
    try:
        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n', ''))
        assert isinstance(da["CoT"], list), "CoT should be list"
        assert da['CoT'][-3]['action'] == 'Inner Thinking', 'Inner Thinking should be the third last action'
        assert da['CoT'][-2]['action'] == 'Final Conclusion', 'Final Conclusion should be the second last action'
        assert da['CoT'][-1]['action'] == 'Verification', 'Verification should be the last action'
        return True, da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False, None


def parse_gpt_response_reformat(response):
    print(response)
    try:
        if '{' != response[0]:
            response = extract_bracket_content(response)
        da = json.loads(response.replace('\n', ''))

        assert isinstance(da["NaturalReasoning"], str), "NaturalReasoning should be str"
        assert '\n' in da["NaturalReasoning"], "NaturalReasoning should have \\n"
        return True, da
    except Exception as e:
        print(e)
        traceback.print_exc()
        return False, None


def get_stream_of_search(longcot):
    temp = '### {}\n{}\n'
    resstr = []
    for x in longcot:
        if 'title' in x:
            resstr.append(temp.format(x['title'], x['content']))
        else:
            resstr.append(temp.format(x['action'].replace('Final Conclusion', 'Conclusion'), x['content']))
    return '\n'.join(resstr).strip()


def filter_data(tmpdata):
    filtered_data = []
    for da in tmpdata:
        if 'Open-ended Verifiable Question' not in da or 'Ground-True Answer' not in da:
            continue
        filtered_data.append(da)

    print(f"Original data size: {len(tmpdata)}, Filtered data size: {len(filtered_data)}")
    return filtered_data


def verify_gpt(conclusion, answer, d):
    query = verify_prompt.format(conclusion, answer)
    response = gpt_instance.retry_call(query)
    d['gpt4_query_cot'].append(query)
    d['gpt4_response_cot'].append(response)
    if 'true' in response.lower():
        d['verify'].append(True)
        return True
    else:
        d['verify'].append(False)
        return False


global wrongtime
wrongtime = 0


def write_piece_order_data(d):
    global wrongtime
    try:
        retry_time = 1
        d['verify'] = []
        d['Long_CoT'] = []
        d['gpt4_query_cot'] = []
        d['gpt4_response_cot'] = []
        d['response_struct'] = []
        d['response_type'] = []
        d['prior_fail_try'] = []

        save_path = os.path.join(save_dir, str(d['process_id']) + ".json")

        # init reason
        # query = query_prompt_init.format(d['Open-ended Verifiable Question'])
        if d['Instruction'] != 'No':
            query = query_prompt_init_with_decompose.format(d['Open-ended Verifiable Question'], d['Instruction'])
        else:
            query = query_prompt_init.format(d['Open-ended Verifiable Question'])
        d['gpt4_query_cot'].append(query)
        for ii in range(retry_time):
            response = gpt_instance.retry_call(query)
            if ii == 0:
                d['gpt4_response_cot'].append(response)
            flag, struct = parse_gpt_response(response)
            if flag:
                d['response_struct'].append(struct["CoT"])
                d['Long_CoT'] = struct["CoT"]
                d['response_type'].append('Init_CoT')
                break
            else:
                print(f'retrying Init_CoT', flush=True)
        if not flag:
            raise Exception('init error')

        verify_gpt(d['Long_CoT'][-2]['content'], d['Ground-True Answer'], d)

        for rethinking_try_time in range(args.max_search_attempts):
            if rethinking_try_time > 0:
                # Archive the failed state
                del d['prior_fail_try']
                save_d['prior_fail_try'].append(d)
                # Replace with a new state
                d = save_d

            # Save the initial state
            save_d = copy.deepcopy(d)

            # Begin search
            for rethink_time in range(args.max_search_depth):
                if d['verify'][-1]:
                    break
                reasoning = json.dumps(d['Long_CoT'][:-1], ensure_ascii=False, indent=2)
                # Search strategy
                if rethink_time > 0:
                    strategy_name, strategy = random.choice(search_strategies)
                else:
                    # exclude Backtracking
                    strategy_name, strategy = random.choice(search_strategies[1:])

                query = strategy.format(d['Open-ended Verifiable Question'], reasoning)
                d['gpt4_query_cot'].append(query)

                for ii in range(retry_time):
                    response = gpt_instance.retry_call(query)
                    flag, struct = parse_gpt_response(response)

                    if flag:
                        d['gpt4_response_cot'].append(response)
                        d['response_struct'].append(struct["CoT"])
                        d['Long_CoT'] = d['Long_CoT'][:-1] + struct["CoT"]
                        d['response_type'].append(f'Re_CoT_{strategy_name}')
                        break
                    else:
                        print(f'retrying strategy {strategy_name}', flush=True)
                if not flag:
                    raise Exception('rethink error')
                verify_gpt(d['Long_CoT'][-2]['content'], d['Ground-True Answer'], d)

            if d['verify'][-1]:
                break

        # If it is still incorrect, generate a final Label_CoT round
        if not d['verify'][-1] and args.efficient_search:
            reasoning = json.dumps(d['Long_CoT'][:-1], ensure_ascii=False, indent=2)
            query = gen_prompt_w_label.format(d['Open-ended Verifiable Question'], reasoning,
                                              d['Ground-True Answer'])
            d['gpt4_query_cot'].append(query)
            for ii in range(retry_time):
                response = gpt_instance.retry_call(query)
                flag, struct = parse_gpt_response(response)
                if flag:
                    d['gpt4_response_cot'].append(response)
                    d['response_struct'].append(struct["CoT"])
                    d['Long_CoT'] = d['Long_CoT'][:-1] + struct["CoT"]
                    d['response_type'].append('Label_CoT')
                    # ignore verify
                    d['verify'].append(True)
                    break
                else:
                    print(f'retrying Label_CoT', flush=True)
            if not flag:
                raise Exception('label error')

        if d['verify'][-1]:
            # Generate complex CoT and final response (Complex_CoT, response)
            sos = get_stream_of_search(d['Long_CoT'])
            query = reformat_to_complex_cot_prompt.format(sos, d['Open-ended Verifiable Question'])
            d['gpt4_query_cot'].append(query)
            for ii in range(retry_time):
                response = gpt_instance.retry_call(query)
                flag, struct = parse_gpt_response_reformat(response)
                if flag:
                    d['gpt4_response_cot'].append(response)
                    d["Complex_CoT"] = struct["NaturalReasoning"]
                    # get response
                    query = get_final_response_prompt.format(d['Complex_CoT'], d['Open-ended Verifiable Question'])
                    d['gpt4_query_cot'].append(query)
                    response = gpt_instance.retry_call(query)
                    d['gpt4_response_cot'].append(response)
                    d["Response"] = response
                    d['Question'] = d['Open-ended Verifiable Question']
                    break

        with open(save_path, mode="w", encoding="utf-8") as fw:
            json.dump(d, fw, ensure_ascii=False, indent=2)
            wrongtime = 0

    except Exception as e:
        traceback.print_exc()
        wrongtime += 1
        if wrongtime > 20:
            assert 1 == 0, 'wrong'
    return 1


def deduplicate_data(data, processed_data):
    processed_ids = {item['process_id'] for item in processed_data}
    return [item for item in data if item['process_id'] not in processed_ids]


def merge_saved_files(save_dir):
    _, _, filenames = [i for i in os.walk(save_dir)][0]
    json_files = [f for f in filenames if f.endswith('.json')]
    res = []
    for file_path in json_files:
        try:
            with open(os.path.join(save_dir, file_path), encoding="utf-8") as f:
                da = json.loads(f.read())
                assert 'Complex_CoT' in da and 'Response' in da
                res.append(da)
        except Exception as e:
            continue
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='training')
    parser.add_argument("--max_search_attempts", type=int, default=2, help="Maximum number of search attempts.")
    parser.add_argument("--max_search_depth", type=int, default=3, help="Maximum search depth.")
    parser.add_argument("--efficient_search", type=bool, default=True, help="Enable efficient search strategy.")
    parser.add_argument("--num_process", type=int, default=5, help="Number of parallel processes.")

    parser.add_argument("--limit_num", type=int, default=5, help="Limit the number of processed items.")
    # parser.add_argument("--limit_num", type=int, help="Limit the number of processed items.")

    args = parser.parse_args()

    # Load dataset - support both JSON and CSV formats
    import os
    dataset_path = "./" + args.dataset
    
    # Check if file exists with .json or .csv extension
    if os.path.exists(dataset_path + ".json"):
        tmpdata = load_dataset("json", data_files={"train": dataset_path + ".json"})
        tmpdata = tmpdata["train"]
        tmpdata = [
            {
                "Open-ended Verifiable Question": record["Open-ended Verifiable Question"],
                "Ground-True Answer": record["Ground-True Answer"],
                "Instruction": record.get("Instruction", "No"),  # Default to "No" if not present
                "Index": record.get("index", idx + 1),
            }
            for idx, record in enumerate(tmpdata)
        ]
    elif os.path.exists(dataset_path + ".csv"):
        tmpdata = load_dataset("csv", data_files={"train": dataset_path + ".csv"})
        tmpdata = tmpdata["train"]
        tmpdata = [
            {
                "Open-ended Verifiable Question": record.get("Question", record.get("Open-ended Verifiable Question", "")),
                "Ground-True Answer": str(record["Ground-True Answer"]),  # Ensure string format
                "Instruction": "No",  # CSV format doesn't have Instruction field
                "Index": idx + 1,
            }
            for idx, record in enumerate(tmpdata)
        ]
    else:
        raise FileNotFoundError(f"Cannot find {dataset_path}.json or {dataset_path}.csv")

    tmp_id = 1
    for da in tmpdata:
        da['process_id'] = tmp_id
        tmp_id += 1
    data = filter_data(tmpdata)

    # if args.limit_num:
    #     data = data[:args.limit_num]

    print(f"read data:{len(data)}")

    task_name = f'{args.dataset}_train_with_CoT'
    save_dir = f'output_deepseek-reasoner_test_fewshot/{task_name}'

    # gpt_instance = GPT()
    # 使用阿里云通义千问模型: qwen-plus (推荐), qwen-turbo (快速), qwen-max (最强)
    gpt_instance = DeepSeek("deepseek-reasoner")
    os.makedirs(save_dir, exist_ok=True)

    # Merge previously processed files
    processed_data = merge_saved_files(save_dir)
    print(f"Previously processed items: {len(processed_data)}")

    input_data = deduplicate_data(data, processed_data)
    print(f"Items remaining for processing: {len(input_data)}")

    with ThreadPoolExecutor(max_workers=args.num_process) as executor:
        list(
            tqdm(executor.map(write_piece_order_data, input_data), total=len(input_data), desc="Processing samples", unit="sample"))

    # for debug
    # results = [write_piece_order_data(item) for item in tqdm(input_data, total=len(input_data))]

    # Merge and save final output
    final_data = merge_saved_files(save_dir)
    output_path = f"./{task_name}.json"
    print(f"Processed {len(final_data)} items. Saving to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(final_data, file, ensure_ascii=False, indent=2)
