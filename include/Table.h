/*
 * Table.h
 *
 *  Created on: Feb 17, 2016
 *      Author: tim
 */

#ifndef TABLE_H_
#define TABLE_H_

#include "leveldb/db.h"
#include <string>
#include <map>

class Table {
public:
	Table(std::string name, std::string path);
	~Table(){};

	template <typename K,typename V> std::map<K,V> get_dictionary(std::string key);

private:
	std::string mypath;
	std::string myname;
	leveldb::DB *db;

};

#endif /* TABLE_H_ */
