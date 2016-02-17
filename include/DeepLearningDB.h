/*
 * DeepLearningDB.h
 *
 *  Created on: Feb 5, 2016
 *      Author: tim
 */

#include "leveldb/db.h"
#include <string>
#include <map>

#ifndef DEEPLEARNINGDB_H_
#define DEEPLEARNINGDB_H_

class Table;

class DeepLearningDB
{
public:
	DeepLearningDB();
	DeepLearningDB(std::string path);
	Table *get_table(std::string tbl);
	~DeepLearningDB(){};

	leveldb::DB *db;
	std::string mypath;
};

#endif /* DEEPLEARNINGDB_H_ */
