# 最基本的数学运算
import tensorflow as tf


def basic_add_variable():
    v1 = tf.Variable(10)
    v2 = tf.Variable(5)
    add = v1 + v2
    sess = tf.Session()
    # Variable必须先初始化
    tf.global_variables_initializer().run(session=sess)
    print("变量10 + 变量5 = " + str(sess.run(add)))


def basic_add_constant():
    c1 = tf.constant(2)
    c2 = tf.constant(3)
    print("常量2 + 常量3 = " + str(tf.Session().run(c1 + c2)))


# 使用placeholder, 节省内存，按需加载
def use_place_holder():
    graph = tf.Graph()
    with graph.as_default():
        value1 = tf.placeholder(dtype=tf.float64)
        value2 = tf.Variable([3, 4], dtype=tf.float64)
        mul = value1 * value2

    with tf.Session(graph=graph) as mySess:
        tf.global_variables_initializer().run()
        value = load_from_remote()
        for partialValue in load_partial(value, 2):
            runResult = mySess.run(mul, feed_dict={value1: partialValue})
            print("乘法(value1, value2) = ", runResult)


def load_from_remote():
    return [-x for x in range(1000)]


def load_partial(value, step):
    index = 0
    while index < len(value):
        yield value[index:index + step]
        index += step
    return


basic_add_variable()
basic_add_constant()
use_place_holder()
