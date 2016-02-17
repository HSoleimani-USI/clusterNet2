/*
 * Table.cpp
 *
 *  Created on: Feb 17, 2016
 *      Author: tim
 */

#include "Table.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

using std::cout;
using std::endl;

Table::Table(std::string name, std::string path)
{
	mypath = path;
	myname = name;


	leveldb::Options options;
	options.create_if_missing = true;
	leveldb::Status status = leveldb::DB::Open(options, path + "/" + name, &db);
}

/*
 *
j.is_null();
j.is_boolean();
j.is_number();
j.is_object();
j.is_array();
j.is_string();
 *
 */

template std::map<std::string,int> Table::get_dictionary(std::string key);
template <typename K,typename V> std::map<K,V> Table::get_dictionary(std::string key)
{
	std::map<K,V> ret;

	std::string value;
    leveldb::Status s = db->Get(leveldb::ReadOptions(), key, &value);
    auto obj = json::parse(value);
    assert(obj.is_object());
	for (json::iterator it = obj.begin(); it != obj.end(); ++it)
	{
	  std::cout << it.key() << " : " << it.value() << "\n";
	  ret[(K)it.key()] = (V)it.value();
	}

	return ret;
}
