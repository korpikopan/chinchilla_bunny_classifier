import tensorflow as tf

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def bunny_or_chinch(image):
    with tf.compat.v1.gfile.FastGFile(image, 'rb') as f:
        #^ Image being read
        data = f.read()
    #^ Data from image file
    # 
    # Loads label file, strips off carriage return
    classes = [line.rstrip() for line in tf.compat.v1.gfile.GFile("rongeurs_labels.txt")]
    # Unpersists graph from file
    
    with tf.compat.v1.gfile.FastGFile("rongeurs_graph.pb", 'rb') as inception_graph:
        definition = tf.compat.v1.GraphDef()
        definition.ParseFromString(inception_graph.read())
        _ = tf.import_graph_def(definition, name='')

    with tf.compat.v1.Session() as session:
        final_result = []
        tensor = session.graph.get_tensor_by_name('final_result:0')
        #^ Feeding data as input and find the first prediction
        result = session.run(tensor, {'DecodeJpeg/contents:0': data})
    
        top_results = result[0].argsort()[-len(result[0]):][::-1] 
        for type in top_results:
            lapin_or_chinchilla = classes[type]
            score = result[0][type]
            print('%-20s : %.5f' % (lapin_or_chinchilla, score))
            final_result.append((lapin_or_chinchilla, score))
    
    return final_result
