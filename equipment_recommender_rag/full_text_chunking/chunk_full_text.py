
# OPEN SCHOLAR CHUNKING
def split_data_into_chunks(text, chunk_sz, min_chunk_sz, keep_last):
    # returns chunks of size <= chunk_sz + min_chunk_sz
    if chunk_sz is None:
        return [text]
    
    text = text.split()
    N = len(text) if keep_last else len(text)-len(text)%chunk_sz
    chunks = [' '.join(text[i:i+chunk_sz]) for i in range(0,N,chunk_sz)]

    if len(chunks) > 1 and len(chunks[-1].split(' ')) < min_chunk_sz:
        # merge the last min_chunk_sz words to the previous chunk
        last_chunk = chunks.pop()
        chunks[-1] += ' ' + last_chunk

    return chunks
