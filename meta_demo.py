from abc import ABCMeta
import logging

import six


def register_class(class_map, class_name, cls):
    assert class_name not in class_map or class_map[class_name] == cls, \
        'confilict class %s , %s is already register to be %s' % (
            cls, class_name, str(class_map[class_name]))
    logging.debug('register class %s' % class_name)
    class_map[class_name] = cls


def get_register_class_meta(class_map, have_abstract_class=True):

    class RegisterABCMeta(ABCMeta):

        def __new__(mcs, name, bases, attrs):
            newclass = super(RegisterABCMeta, mcs).__new__(mcs, name, bases, attrs)
            print("name: {}".format(name))
            print("bases: {}".format(bases))
            print("attrs: {}".format(attrs))
            register_class(class_map, name, newclass)

            # setattr(newclass, 'create_class', mcs.create_class)
            print(type(newclass))
            return newclass

        @classmethod
        def create_class(mcs, name):
            if name in class_map:
                return class_map[name]
            else:
                raise Exception('Class %s is not registered. Available ones are %s' %
                                (name, list(class_map.keys())))

    return RegisterABCMeta


_meta_class_map = {}
meta_class = get_register_class_meta(_meta_class_map)


class ImplementClass(six.with_metaclass(meta_class, object)):
    def __init__(self, a):
        self.a = a


if __name__ == '__main__':
    imp = ImplementClass.create_class(ImplementClass.__name__)(19)
    print(imp.__dict__)

