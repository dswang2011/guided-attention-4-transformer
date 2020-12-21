from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_IMDB_data():
    le = preprocessing.LabelEncoder()
    texts, labels = [], []
    file_path = '../datasets/IMDB/IMDB.csv'
    with open(file_path, 'r', encoding='utf8') as fr:
        content = [x.rstrip() for x in fr.readlines()]
        for line in content[1:]:
            # line = self.processed_text(line)
            strs = line.strip().rsplit(',', 1)
            texts.append(strs[0])
            labels.append(strs[1])
    encoded_labels = le.fit_transform(labels)
    return texts, encoded_labels


def load_ROTTENTOMATOES_data():
    le = preprocessing.LabelEncoder()
    texts, labels = [], []
    file_path = '../datasets/ROTTENTOMATOES/rottentomatoes.txt'
    with open(file_path, 'r', encoding='utf8') as fr:
        content = [x.rstrip() for x in fr.readlines()]
        for line in content[1:]:
            # line = self.processed_text(line)
            strs = line.strip().rsplit(',', 1)
            texts.append(strs[0])
            labels.append(strs[1])
    encoded_labels = le.fit_transform(labels)
    return texts, encoded_labels


def split_data(folder_dest, dataset, t_size):
    X, y = [], []
    if dataset == "IMDB":
        X,y = load_IMDB_data()
    elif dataset == "ROTTENTOMATOES":
        X, y = load_IMDB_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size, random_state = 42)

    fout_train = open(folder_dest+'train.csv','w',encoding='utf-8')
    fout_test = open(folder_dest+'test.csv','w',encoding='utf-8')

    for x_train, y_train in zip(X_train,y_train):
        print(y_train, x_train, sep=',', file=fout_train)
    for x_test, y_test in zip(X_test,y_test):
        print(y_test, x_test, sep=',', file=fout_test)

    fout_train.close()
    fout_test.close()


if __name__ == '__main__':
    # split_data(folder_dest="../datasets/IMDB/", dataset='IMDB', t_size=0.20)
    split_data(folder_dest="../datasets/ROTTENTOMATOES/", dataset='ROTTENTOMATOES', t_size=0.20)
