# 启动测试命令 pytest . -s -v
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.nodes.retriever import retriever


def test_document_grader_answer_yes() -> None:
    question = "agent memory"
    documents = retriever.invoke(question)
    first_doc_content = documents[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "documents": first_doc_content}
    )

    assert res.binary_score == "yes"
