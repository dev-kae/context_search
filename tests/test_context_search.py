import pytest
from entities.context_search import ContextSearch


def test_tokenization():
    text = "This is a test sentence."
    cs = ContextSearch(text)
    tokens = cs.tokens

    expected_results = ["This", "is", "a", "test", "sentence", "."]
    assert tokens == expected_results


def test_sentence_splitter():
    text = "This is a test sentence. This is another sentence."
    cs = ContextSearch(text)
    sentences = cs.sentences

    expected_results = ["This is a test sentence.", "This is another sentence."]
    assert sentences == expected_results


def test_context_search():
    text = "Python maybe slow. Java is verbose."
    cs = ContextSearch(text)
    result = cs.search("slow")
    expected_result = ["Python maybe slow."]

    assert result == expected_result


def test_context_search_if_no_representant_found():
    text = "Python maybe slow. Java is verbose."
    cs = ContextSearch(text)
    result = cs.search("slo")
    expected_result = []

    assert result == expected_result



def test_return_for_no_context_search_representant_found():
    text = "Python maybe slow. Java is verbose."
    cs = ContextSearch(text)
    result = cs.search("slo")
    expected_result = []

    assert result == expected_result


def test_advanced_context_search():
    text = """
    Viajar é uma das melhores maneiras de conhecer novas culturas e expandir horizontes. Destinos como Paris, Tóquio e Nova York oferecem experiências únicas.
    No entanto, é importante planejar bem os gastos para evitar surpresas financeiras durante a viagem.
    
    A inteligência artificial está revolucionando diversas indústrias. Empresas estão usando algoritmos avançados para analisar grandes volumes de dados e tomar
    decisões mais rápidas e precisas. No entanto, há preocupações éticas sobre o uso indevido dessas tecnologias, especialmente no campo da privacidade dos usuários.
    """
    cs = ContextSearch(text)
    result = cs.advanced_search("Paris")
    expected_result = [
        'Destinos como Paris, Tóquio e Nova York oferecem experiências únicas.'
    ]
    
    assert result == expected_result
