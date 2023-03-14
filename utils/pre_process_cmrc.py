import argparse
import json


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--input_file", default=None, type=str,
                        help="Path of the input file.")
    parser.add_argument("--output_file", default=None, type=str,
                        help="Path of the output file.")

    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file
    obj = json.load(open(input_file, "r", encoding="utf-8"))

    src, tgt, ids = [], [], []

    for data in obj["data"]:
        for paragraph in data["paragraphs"]:
            context = paragraph["context"].replace("\n", "")
            for qas in paragraph["qas"]:
                id = qas["id"]
                question = qas["question"].replace("\n", "")
                answer = qas["answers"][0]["text"].replace("\n", "")
                contextual = question + context
                if id and contextual and answer:
                    src.append(contextual)
                    tgt.append(answer)
                    ids.append(id)
                    print(len(contextual))

    fos = open(output_file + ".source", "w", encoding="utf-8")
    fot = open(output_file + ".target", "w", encoding="utf-8")
    fid = open(output_file + ".ids", "w", encoding="utf-8")

    for s in src:
        fos.write("%s\n" % s)
    for t in tgt:
        fot.write("%s\n" % t)
    for i in ids:
        fid.write("%s\n" % i)


if __name__ == '__main__':
    main()
