# 启动测试命令 pytest . -s -v
from pprint import pprint

from langsmith import traceable

from graph.chains.answer_grader import GradeAnswer, answer_grader
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucinations,
                                               hallucination_grader)
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.nodes.retriever import retriever


@traceable(name="test_document_grader_answer_yes")
def test_document_grader_answer_yes() -> None:
    question = "agent memory"
    documents = retriever.invoke(question)
    first_doc_content = documents[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "documents": first_doc_content}
    )

    assert res.binary_score == "yes"


@traceable(name="test_document_grader_answer_no")
def test_document_grader_answer_no() -> None:
    question = "agent memory"
    documents = retriever.invoke(question)
    first_doc_content = documents[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "documents": first_doc_content}
    )

    assert res.binary_score == "no"


@traceable(name="test_generation_chain")
def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


@traceable(name="test_hallucination_grader_answer_yes")
def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})

    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


@traceable(name="test_hallucination_grader_answer_no")
def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score


@traceable(name="test_answer_grader_answer_yes")
def test_answer_grader_answer_yes() -> None:

    res: GradeAnswer = answer_grader.invoke(
        {
            "question": "1 + 1 = ?",
            "generation": "1加1等于2",
        }
    )
    assert res.binary_score
