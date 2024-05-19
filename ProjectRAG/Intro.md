# What is RAG (Retrival-Augmented Generation)?

In RAG we retrieve information from an external knowledge base and give that information to LLM. The external knowledge base is our window into the worl beyond the LLM's training data.

# What is Chunkning?

Chunking is the process of breaking down large pieces of text into smaller segments. It's an essential technique that helps optimize the relevance of the content we get back from a vector database.

as a rule of thumb, if a chunk of text makes sense without the surrounding context to a human, it will make sense to the language model as well.

findind the optimal chunk size for the documents in the corpus is curcial to ensure that the search results are accurate and relevant.
