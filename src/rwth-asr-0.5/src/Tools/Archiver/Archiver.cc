// Copyright 2011 RWTH Aachen University. All rights reserved.
//
// Licensed under the RWTH ASR License (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <ext/functional>
#include <Core/Application.hh>
#include <Core/Archive.hh>
#include <Core/Choice.hh>
#include <Core/CompressedStream.hh>
#include <Core/Directory.hh>
#include <Core/FileArchive.hh>
#include <Core/Parameter.hh>
#include <Core/StringUtilities.hh>
#include <Core/TextStream.hh>

using namespace Core;
using __gnu_cxx::select2nd;

enum Mode { Add, Combine, Copy, Extract, ExtractAll, List, Recover, Remove };
static const Choice modeChoice_(
    "add",     Add,
    "combine", Combine,
    "copy",    Copy,
    "extract", Extract,
    "extractAll", ExtractAll,
    "list",    List,
    "recover", Recover,
    "remove",  Remove,
    Choice::endMark());
static const ParameterChoice p_mode_("mode", &modeChoice_, "mode", List);
static const ParameterBool p_verbose_("verbose", "verbose mode", false);
static const ParameterBool p_compress_("compress", "compress files", false);
static const ParameterString p_select_("select", "select only entries from file", "");
static const ParameterBool p_quiet_("quiet", "less output", false);



enum OverwriteMode {
    overwriteKeepFirst,
    overwriteReplace,
    overwriteCheckEquality };
static const Choice overwriteModeChoice(
    "no",         overwriteKeepFirst,
    "keep-first", overwriteKeepFirst,
    "yes",        overwriteReplace,
    "replace",    overwriteReplace,
    "check",      overwriteCheckEquality,
    "save",       overwriteCheckEquality,
    Core::Choice::endMark());
static const ParameterChoice paramOverwrite(
    "overwrite",
    &overwriteModeChoice,
    "what to do when an archive member already exists with the same name",
    overwriteCheckEquality);
static const ParameterString paramSelect(
    "select",
    "select only entries from file; only valid for combine",
    "");


typedef std::vector<std::pair<std::string, bool> > Selection;
typedef std::vector<std::string> StringVector;


class ArchiverApplication : public Core::Application {
private:
    Mode mode_;
    bool verbose_;
    bool quiet_;
    bool compress_;
    OverwriteMode overwrite_;

private:
    std::string getUsage() const {
	std::string usage = "manipulate SPRINT archives\n"
			    "\n"
			    "usage: archiver [OPTION] <archive> <FILE>...\n"
			    "\n"
			    "options:\n"
			    "   --compress <bool>\tcompress new files added to the archive\n"
			    "   --mode <mode>\tchoose operational mode (see below for available modes)\n"
			    "   --verbose <bool>\tbe a bit more verbose\n"
			    "   --quiet <bool>\tless output\n"
			    "   --select <file>\tapply operation only to files listed in <file>; only supported by combine and copy\n"
			    "   --overwrite <mode>\twhat to do when archive member already exists"
			    "\n"
			    "modes:\n"
			    "   add\t\tadd files or directories to archive\n"
			    "   combine\tcombine other archives into new one\n"
			    "   copy\t\tcopy files between archives directly (option 'compress' is ignored)\n"
			    "   extract\textract single files with path\n"
			    "   extractAll\textract all files to given directory\n"
			    "   list\t\tlist archive(s) (default)\n"
			    "   remove\tremove single files from archive\n"
			    "   recover\trecover archive (if internal structure is broken)\n"
			    "\n"
			    "overwrite-modes:\n"
			    "   no\t\tno overwriting\n"
			    "   yes\t\toverwrite files with the same name\n"
			    "   check\tcheck for data equality of archive members with the same name\n"
			    "\n";
	return usage;
    }

public:
    ArchiverApplication() : mode_(List), verbose_(false), compress_(false) {
	setTitle("archiver");
	setDefaultLoadConfigurationFile(false);
    }

    bool addFile(Archive *archive, std::istream &src, const std::string &name) {
	respondToDelayedErrors();
	if (archive->hasFile(name)) {
	    std::cout << "    file \"" << name << "\" already exists: ";
	    switch (overwrite_) {
	    case overwriteKeepFirst: {
		std::cout << "will keep existing file" << std::endl;
		return true;
	    } break;
	    case overwriteReplace: {
		std::cout << "will replace existing file" << std::endl;
		return writeFileToArchive(archive, src, name);
	    } break;
	    case overwriteCheckEquality: {
		bool isEqual = compareFileToArchive(archive, src, name);
		std::cout << ((isEqual)
			      ? "files are equal"
			      : "FILES DIFFER") << std::endl;
		return isEqual;
	    } break;
	    default: defect();
	    }
	} else {
	    return writeFileToArchive(archive, src, name);
	}
    }

    bool compareFileToArchive(Archive *archive, std::istream &src, const std::string &name) {
	Core::ArchiveReader ref(*archive, name);
	if (!ref.isOpen())
	    error("could not open member \"%s\" in archive for reading", name.c_str());
	while (!src.eof()) {
	    char buf1[1024], buf2[1024];
	    src.read(buf1, 1024);
	    ref.read(buf2, 1024);
	    if (src.gcount() != ref.gcount()) return false;
	    if (memcmp(buf1, buf2, src.gcount())) return false;
	}
	if (!ref.eof()) return false;
	return true;
    }

    bool writeFileToArchive(Archive *a, std::istream &src, const std::string &archiveName)
    {
	Core::ArchiveWriter dest(*a, archiveName, compress_);
	if (!dest.isOpen())
	    error("could not open member \"%s\" in archive for writing", archiveName.c_str());
	while (!src.eof()) {
	    char buf[1024];
	    src.read(buf, 1024);
	    dest.write(buf, src.gcount());
	}
	return dest.good();
    }

    void addFile(Archive *a, const std::string &name, const std::string &archiveName) {
	std::ifstream src(name.c_str());
	if (!src)
	    error("could not open file \"%s\" for reading", name.c_str());
	else if (!addFile(a, src, archiveName))
	    error("could not add file \"%s\" to archive", name.c_str());
    }

    void addDirectory(Archive *a, const std::string &name) {
	for (Core::DirectoryFileIterator it(name, &DirectoryFileIterator::fileFilter); it; ++it) {
	    std::cout << "  adding file " << it.path() << std::endl;
	    addFile(a, joinPaths(it.base(), it.path()), it.path());
	}
    }


    bool copyAllFiles(Archive *a, const std::string &name) {
	Archive *src = Core::Archive::create(config, name, Core::Archive::AccessModeRead);
	if (!src) {
	    error("could not open archive '%s'", name.c_str());
	    return false;
	}
	bool err = false;
	for (Archive::const_iterator i = src->files(); i; ++i) {
	    if (!a->copyFile(*src, i.name())) {
		err = true;
		std::cout << i.name() << ": could not copy file to archive" << std::endl;
	    }
	}
	delete src;
	if (err) {
	    error("an error has occurred during copy");
	}
	return true;
    }

    bool copySelectedFiles(Archive *a, StringVector::const_iterator namesBegin, StringVector::const_iterator namesEnd,
			   Selection &selection) {
	for (StringVector::const_iterator name = namesBegin; name != namesEnd; ++name) {
	    Archive *src = Core::Archive::create(config, *name, Core::Archive::AccessModeRead);
	    if (!src) {
		error("could not open archive '%s'", name->c_str());
		continue;
	    }
	    u32 count = 0;
	    for (Selection::iterator i = selection.begin(); i != selection.end(); ++i) {
		if (verbose_) std::cout << i->first << "\t";
		if (!i->second) {
		    count += (i->second = a->copyFile(*src, i->first));
		    if (verbose_) std::cout << (i->second ? "OK" : "not found") << std::endl;
		} else {
		    if (verbose_) std::cout << "already copied" << std::endl;
		}
	    }
	    std::cout << "copied " << count << " files from " << *name << std::endl;

	    delete src;
	}
	u32 missing = selection.size() - std::count_if (selection.begin(), selection.end(),
							    select2nd<Selection::value_type>());
	if (missing) {
	    for (Selection::iterator i = selection.begin(); i != selection.end(); ++i) {
		if (!i->second) {
		    std::cout << "missing file: " << i->first << std::endl;
		}
	    }
	    error("not all files have been copied");
	}
	return true;
    }


    bool addArchive(Archive *a, const std::string &name) {
	Archive *src = Core::Archive::create(config, name, Core::Archive::AccessModeRead);
	if (src) {
	    for (Archive::const_iterator i = src->files(); i; ++i) {
		Core::ArchiveReader reader(*src, i.name());
		if (reader.isOpen()) {
		    std::cout << "  adding file " << i.name() << std::endl;
		    if (!addFile(a, reader, i.name()))
			std::cout << i.name() << ": could not add file to archive" << std::endl;
		} else
		    error("could not open file '%s' in archive %s for reading", i.name().c_str(), name.c_str());
	    }
	    delete src;
	}
	return true;
    }

    bool extractFile(Archive *a, const std::string &name, const std::string outputName = "") {
	Core::ArchiveReader src(*a, name);
	if (!src.isOpen())
	    error("could not open file '%s' in archive %s for reading", name.c_str(), name.c_str());
	respondToDelayedErrors();
	std::string targetName = (outputName.empty() ? name : outputName);
	if (targetName.size() > 3 && targetName.substr(targetName.size()-3) == ".gz")
	    targetName = targetName.substr(0, targetName.size()-3);
	if (!isDirectory(directoryName(targetName).c_str()))
	    createDirectory(directoryName(targetName).c_str());
	std::ofstream dest(targetName.c_str());
	if (!dest)
	    error("could not create file '%s' for writing", targetName.c_str());
	while (!src.eof()) {
	    char buf[1024];
	    src.read(buf, 1024);
	    dest.write(buf, src.gcount());
	}
	return true;
    }

    void loadSelection(const std::string &filename, Selection &selection) {
	verify(!filename.empty());
	Core::CompressedInputStream *cis = new Core::CompressedInputStream(filename.c_str());
	Core::TextInputStream is(cis);
	if (!is)
	    criticalError("Failed to open selection file \"%s\".", filename.c_str());
	std::string s;
	while (Core::getline(is, s) != EOF) {
	    if ((s.size() == 0) || (s.at(0) == '#'))
		continue;
	    Core::normalizeWhitespace(s);
	    Core::suppressTrailingBlank(s);
	    selection.push_back(std::make_pair(s, false));
	}
    }

    int main(const std::vector<std::string> &arguments) {
	Archive *a;
	verbose_ = p_verbose_(config);
	quiet_ = p_quiet_(config);
	compress_ = p_compress_(config);
	overwrite_ = OverwriteMode(paramOverwrite(config));
	if (overwrite_ == overwriteReplace) {
	    config.set("*.allow-overwrite", "true");
	}
	mode_ = Mode(p_mode_(config));
	switch(mode_) {
	case Add:
	    a = Core::Archive::create(config, arguments[0]);
	    if (a) {
		for (u32 i = 1; i < arguments.size(); i++) {
		    struct stat buf;
		    if (stat(arguments[i].c_str(), &buf) == 0) {
			if (S_ISREG(buf.st_mode)) {
			    std::cout << "adding file " << arguments[i] << std::endl;
			    addFile(a, arguments[i], baseName(arguments[i]));
			} else if (S_ISDIR(buf.st_mode)) {
			    std::cout << "adding directory " << arguments[i] << std::endl;
			    addDirectory(a, arguments[i]);
			}
		    } else {
			error("Could not find file \"%s\".", arguments[i].c_str());
		    }
		}
		delete a;
	    }
	    break;
	case Combine:
	    a = Core::Archive::create(config, arguments[0]);
	    if (a) {
		std::string selectFile = paramSelect(config);
		if (selectFile.empty()) {
		    for (u32 i = 1; i < arguments.size(); i++) {
			std::cout << "adding contents from archive " << arguments[i] << std::endl;
			addArchive(a, arguments[i]);
		    }
		} else {
		    Selection selection;
		    loadSelection(selectFile, selection);
		    s32 n = selection.size();
		    std::cout << "selection contains " << n << " files" << std::endl;
		    for (u32 i = 1; (i < arguments.size()) && (n > 0); i++) {
			std::cout << "adding selected content from archive " << arguments[i] << std::endl;
			Archive *srcA = Core::Archive::create(config, arguments[i], Core::Archive::AccessModeRead);
			if (srcA) {
			    for (Selection::iterator it = selection.begin(); (it != selection.end()) && (n > 0); ++it) {
				if (!it->second && srcA->hasFile(it->first)) {
				    Core::ArchiveReader reader(*srcA, it->first);
				    if (reader.isOpen()) {
					std::cout << "  adding file " << it->first << std::endl;
					if (addFile(a, reader, it->first))
					    { it->second = true; --n; }
					else
					    std::cout << it->first << ": could not add file to archive" << std::endl;
				    } else
					error("could not open file '%s' in archive %s for reading", it->first.c_str(), arguments[i].c_str());
				}
			    }
			    delete srcA;
			}
		    }
		    if (n > 0) {
			std::cout << "could not find " << n << " files:" << std::endl;
			for (Selection::const_iterator it = selection.begin(); (it != selection.end()) && (n > 0); ++it)
			    if (!it->second)
				std::cout << "  missing file " << it->first << std::endl;
		    }
		}
		delete a;
	    }
	    break;
	case Copy:
	    if (arguments.size() < 2) {
		error("no souce archive given");
	    } else {
		std::string selectionFile = paramSelect(config);
		a = Core::Archive::create(config, arguments[0]);
		if (a) {
		    if (selectionFile.empty()) {
			for (u32 i = 1; i < arguments.size(); ++i) {
			    std::cout << "copy all files from " << arguments[i] << " to " << arguments[0] << std::endl;
			    copyAllFiles(a, arguments[i]);
			}
		    } else {
			Selection selection;
			loadSelection(selectionFile, selection);
			copySelectedFiles(a, arguments.begin() + 1, arguments.end(), selection);
		    }
		    delete a;
		}
	    }
	    break;
	case Extract:
	    a = Core::Archive::create(config, arguments[0]);
	    if (a) {
		for (u32 i = 1; i < arguments.size(); i++) {
		    std::cout << "extracting file " << arguments[i] << std::endl;
		    extractFile(a, arguments[i]);
		}
		delete a;
	    }
	    break;
	case ExtractAll:
	    a = Core::Archive::create(config, arguments[0]);
	    if (a) {
		std::string prefix = "./";
		if (arguments.size() > 1)
		    prefix = arguments[1];
		for (Archive::const_iterator i = a->files(); i; ++i) {
		    std::string output = prefix + i.name();
		    std::cout << "extracting file " << i.name() << " to " << output << std::endl;
		    extractFile(a, i.name(), output);
		}
		delete a;
	    }
	    break;
	case List:
	    for (u32 i = 0; i < arguments.size(); i++) {
		Archive *a = Core::Archive::create(config, arguments[i], Archive::AccessModeRead);
		if (a) {
		    if (quiet_) {
			for (Archive::const_iterator i = a->files(); i; ++i)
			    std::cout << i.name() << std::endl;
		    } else {
			std::cout << std::endl << "archive: " << arguments[i] << std::endl;
			std::cout << *a;
		    }
		    delete a;
		}
	    }
	    break;
	case Recover:
	    a = Core::Archive::create(config, arguments[0], Archive::AccessModeWrite);
	    if (!a->recover()) {
		log("recovery failed");
	    } else {
		log("recovery successfull");
	    }
	    break;
	case Remove:
	    a = Core::Archive::create(config, arguments[0]);
	    if (a) {
		for (u32 i = 1; i < arguments.size(); ++i) {
		    if (!a->removeFile(arguments[i])) {
			error("cannot remove \"%s\"", arguments[i].c_str());
		    }
		}
		delete a;
	    }
	    break;
	default:
	    break;
	}
	return EXIT_SUCCESS;
    }
};

APPLICATION(ArchiverApplication)
