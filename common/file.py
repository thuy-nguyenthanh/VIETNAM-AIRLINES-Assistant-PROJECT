import codecs
import os

def get_parentDir():
    fileDir = os.path.dirname(os.path.abspath(__file__))
    parentDir = os.path.dirname(fileDir)
    return parentDir


def get_dbtrain():
    filename = get_parentDir() + '\data\Questions.csv'
    f = codecs.open(filename, 'rU', 'utf-8')  # Read Unicode text
    db_train = []
    for line in f:
        data = line.split("|")
        db_train.append({"Question": data[1].replace("\r\n", ""), "Intent": int(data[0].replace("\ufeff", ""))})
    return db_train


def get_dbtrain_extend():
    filename = get_parentDir() + '\data\Questions_Extend.csv'
    f = codecs.open(filename, 'rU', 'utf-8')  # Read Unicode text
    db_train_extend = []
    for line in f:
        data = line.split("|")
        db_train_extend.append({"Question": data[1].replace("\r\n", ""), "Intent": int(data[0].replace("\ufeff", ""))})
    return db_train_extend


def get_dbanswers():
    filename = get_parentDir() + '\data\Answers.csv'
    f = codecs.open(filename, 'rU', 'utf-8')  # Read Unicode text
    db_answers = []
    for line in f:
        data = line.split("|")
        #db_answers.append(data[1].replace("\r\n", ""))
        db_answers.append({"Answers": data[1].replace("\r\n", ""), "Intent": int(data[0].replace("\ufeff", ""))})
    return db_answers


def get_fallback_intent():
    fallback_intent = ["Xin lỗi! tôi không hiểu ý của bạn, hãy nêu câu hỏi đầy đủ hơn.",
                       "Vui lòng mô tả đầy đủ thông tin, để tôi có thể tìm câu trả lời phù hợp nhất!",
                       "Tôi vẫn chưa hiểu được câu hỏi của bạn, vui lòng mô tả đầy đủ hơn nhé!",
                       "Tôi chưa hiểu câu hỏi này, có thể mô tả đầy đủ thông tin hoặc tôi sẽ gửi câu hỏi này đến Phòng CSKH để hỗ trợ bạn!"]
    return fallback_intent

