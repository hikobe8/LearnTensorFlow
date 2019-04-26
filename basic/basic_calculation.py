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


basic_add_variable()
basic_add_constant()
