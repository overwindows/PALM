import sys
import os
import hashlib
import struct
import subprocess
import collections


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote,
              dm_double_close_quote, ")", ' ', '\t']  # acceptable ways to end a sentence

wiki_train_raw = "wiki.train.raw"
wiki_val_raw = "wiki.valid.raw"
wiki_test_raw = "wiki.test.raw"

finished_files_dir = "/datadrive/wikitext"

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
# num_expected_cnn_stories = 92579
# num_expected_dm_stories = 219506


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            if line.strip() and not line.strip().startswith('= = =') and len(line.strip()) > 16:
                lines.append(line.strip())
    return lines


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode())
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


# def fix_missing_period(line):
#     """Adds a period to a line that is missing a period"""
#     if "@highlight" in line:
#         return line
#     if line == "":
#         return line
#     if line[-1] in END_TOKENS:
#         return line
#     # print line[-1]
#     return line + " ."


def get_src_tgt(line):
    # lines = read_text_file(story_file)

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    # lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    # article_lines = []
    # highlights = []
    # next_is_highlight = False
    # for idx,line in enumerate(lines):
    #   if line == "":
    #     continue # empty line
    #   elif line.startswith("@highlight"):
    #     next_is_highlight = True
    #   elif next_is_highlight:
    #     highlights.append(line)
    #   else:
    #     article_lines.append(line)

    # # Make article into a single string
    # article = ' '.join(article_lines)

    # # Make abstract into a signle string
    # abstract = ' '.join(highlights)
    idx = int(len(line) * 0.8)
    for i in range(idx, -1, -1):
        if line[i] in END_TOKENS:
          idx = i
          break
    assert idx > 0, line
    return line[:idx], line[idx:]


def write_to_bin(input_file, out_prefix):
    """Reads the .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    print("Making bin file for Wikipedia listed in %s..." % input_file)
    wiki_list = read_text_file(input_file)
    # url_hashes = get_url_hashes(url_list)
    # story_fnames = [s+".story" for s in url_hashes]
    num_wikis = len(wiki_list)

    with open(out_prefix + '.source', 'wt') as source_file, open(out_prefix + '.target', 'wt') as target_file:
        for idx, s in enumerate(wiki_list):
            if idx % 1000 == 0:
                print("Writing wiki %i of %i; %.2f percent done" %
                      (idx, num_wikis, float(idx)*100.0/float(num_wikis)))

            # # Look in the story dirs to find the .story file corresponding to this url
            # if os.path.isfile(os.path.join(cnn_stories_dir, s)):
            #   story_file = os.path.join(cnn_stories_dir, s)
            # elif os.path.isfile(os.path.join(dm_stories_dir, s)):
            #   story_file = os.path.join(dm_stories_dir, s)
            # else:
            #   print("Error: Couldn't find story file %s in either story directories %s and %s." % (s, cnn_stories_dir, dm_stories_dir))
            #   # Check again if stories directories contain correct number of files
            #   print("Checking that the stories directories %s and %s contain correct number of files..." % (cnn_stories_dir, dm_stories_dir))
            #   check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
            #   check_num_stories(dm_stories_dir, num_expected_dm_stories)
            #   raise Exception("Stories directories %s and %s contain correct number of files but story file %s found in neither." % (cnn_stories_dir, dm_stories_dir, s))

            # Get the strings to write to .bin file
            source, target = get_src_tgt(s)

            # Write article and abstract to files
            source_file.write(source + '\n')
            target_file.write(target + '\n')

    print("Finished writing files")


def check_num_stories(stories_dir, num_expected):
    num_stories = len(os.listdir(stories_dir))
    if num_stories != num_expected:
        raise Exception("stories directory %s contains %i files but should contain %i" % (
            stories_dir, num_stories, num_expected))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python3 create_pretraining_data.py <wikitext_raw_dir>")
        sys.exit()
    wikitext_raw_dir = sys.argv[1]

    # Check the stories directories contain the correct number of .story files
    # check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
    # check_num_stories(dm_stories_dir, num_expected_dm_stories)

    # Create some new directories
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # Read the wikipedias, do a little postprocessing then write to bin files
    write_to_bin(os.path.join(wikitext_raw_dir, wiki_test_raw),
                 os.path.join(finished_files_dir, "test"))
    write_to_bin(os.path.join(wikitext_raw_dir, wiki_val_raw),
                 os.path.join(finished_files_dir, "val"))
    write_to_bin(os.path.join(wikitext_raw_dir, wiki_train_raw),
                 os.path.join(finished_files_dir, "train"))
