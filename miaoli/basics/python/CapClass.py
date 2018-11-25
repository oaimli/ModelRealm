#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2018/9/3 21:11
# @Author  : Miao Li
# @File    : CapClass.py


class Employee(object):
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
    EMP_COUNT = 0

    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.EMP_COUNT += 1

    def displayCount(self):
        """Fetches rows from a Bigtable.

        Retrieves rows pertaining to the given keys from the Table instance
        represented by big_table.  Silly things may happen if
        other_silly_variable is not None.

        Args:
            big_table: An open Bigtable Table instance.
            keys: A sequence of strings representing the key of each table row
                to fetch.
            other_silly_variable: Another optional variable, that has a much
                longer name than the other args, and which does nothing.

        Returns:
            A dict mapping keys to the corresponding table row data
            fetched. Each row is represented as a tuple of strings. For
            example:

            {'Serak': ('Rigel VII', 'Preparer'),
             'Zim': ('Irk', 'Invader'),
             'Lrrr': ('Omicron Persei 8', 'Emperor')}

            If a key from the keys argument is missing from the dictionary,
            then that row was not found in the table.

        Raises:
            IOError: An error occurred accessing the bigtable.Table object.
        """
        print("Total Employee %d" % Employee.EMP_COUNT)

    def displayEmployee(self):
        print("Name : ", self.name, ", Salary: ", self.salary)
