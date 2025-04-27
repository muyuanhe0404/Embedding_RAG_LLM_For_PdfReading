from qa_rag import retrieve, answer_rag


def test_retrieve():
    print("\n test index (retrieve) ")
    docs = retrieve("Who is Gatsby?", top_k=3)
    for i, doc in enumerate(docs, 1):
        print(f" chunk {i} ")
        print(doc[:200].replace("\n", " "), "...\n")


def test_answer():
    print("\n test generate (answer_rag) ")
    answer = answer_rag("Who is Gatsby ?", top_k=5, max_length=128)
    print(answer)


if __name__ == '__main__':
    test_retrieve()
    test_answer()
