from vggish.vggish_result import extract_vggish_embeddings
def audioRun(audio):
    print(audio.shape)
    return extract_vggish_embeddings(audio)