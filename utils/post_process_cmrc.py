import argparse
import json


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--predict_file", default=None, type=str,
                        help="Path of the input file.")
    parser.add_argument("--sid_file", default=None, type=str,
                        help="Path of the input file.")
    parser.add_argument("--output_file", default=None, type=str,
                        help="Path of the output file.")

    args = parser.parse_args()

    predict_file = args.predict_file
    sid_file = args.sid_file
    output_file = args.output_file
    fip = open(predict_file, "r", encoding="utf-8")
    fis = open(sid_file, "r", encoding="utf-8")

    id_map = [line.strip() for line in fis.readlines()]

    answers = {} 

    for index, line in enumerate(fip.readlines()):
        if index % 5 != 1:
            continue
        pairs = line.strip().split("\t")
        if len(pairs) != 2:
            continue
        print(str(index) + "\t" + line)
        sid, generated_answer = int(pairs[0].split("-")[1]), pairs[1]
        qid = id_map[sid]
        answers[qid] = generated_answer

    json.dump(answers, open(output_file, "w", encoding="utf-8"), ensure_ascii=False, indent=True)

if __name__ == '__main__':
    main()
