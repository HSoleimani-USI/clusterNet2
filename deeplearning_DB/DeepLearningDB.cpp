#include <DeepLearningDB.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <Table.h>



DeepLearningDB::DeepLearningDB()
{

	struct passwd *pw = getpwuid(getuid());

	const char *homedir = pw->pw_dir;
	leveldb::Options options;
	options.create_if_missing = true;
	mypath = std::string(homedir) + "/.nlpdb";
	leveldb::Status status = leveldb::DB::Open(options,mypath , &db);

}

DeepLearningDB::DeepLearningDB(std::string path)
{
	leveldb::Options options;
	options.create_if_missing = true;
	leveldb::Status status = leveldb::DB::Open(options, path, &db);
	mypath = path;
}

Table *DeepLearningDB::get_table(std::string tbl)
{
	return new Table(tbl, mypath);
}

