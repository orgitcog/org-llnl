package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func main() {
	// Handle command line arguments
	var winePath, winePrefix string
	flag.StringVar(&winePath, "winepath", "", "the root path where wine is installed")
	flag.StringVar(&winePrefix, "wineprefix", "/usr/workspace/wsb/cyme/wine32_prefix", "the wine prefix to use")
	var hasplmPath string
	flag.StringVar(&hasplmPath, "hasplmpath", "/usr/workspace/wsb/cyme/sentinel_lm", "the path to the hasplm daemon")
	var hasplmServer string
	flag.StringVar(&hasplmServer, "hasplmserver", "localhost", "remote hasplm server to use (eg: eng-tools.example.com)")
	var cymePath string
	flag.StringVar(&cymePath, "cymepath", "C:\\Program Files\\CYME\\CYME", "path to the cyme installation")
	var pythonScript string
	flag.StringVar(&pythonScript, "python", "", "python script to run")
	flag.Parse()

	// pass additional arguments to command run

	launcherDir, err := os.Executable()
	launcherDir = filepath.Dir(launcherDir)
	fmt.Println(launcherDir)

	// Make sure wine command is available
	// Make sure hasplmd is also available
	// Make sure CYME executable path is known
	// Check if told to run python script or not, and if python argument is given

	// Try to connect to license server
	features, err := GetSentinelFeatures()
	if err != nil {
		// create tmp directory with a basic hasplmd ini file with the remote server configured
		haspTmpDir, err := ioutil.TempDir("", "hasplm_prefix")
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(haspTmpDir)
		defer os.RemoveAll(haspTmpDir)

		haspIni, err := os.Create(filepath.Join(haspTmpDir, "hasplm.ini"))
		if err != nil {
			log.Fatal(err)
		}

		hostname, err := os.Hostname()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Fprintf(haspIni, haspConfigContents, hostname, hasplmServer)

		// HASPLM_PREFIX=$HOME/hasplmd_prefix sentinel_lm/hasplmd -s
		os.Setenv("HASPLM_PREFIX", haspTmpDir)

		// TODO try to autodetect hasplmd bin directory if not given
		haspCmd := exec.Command(filepath.Join(hasplmPath, "hasplmd"), "-s")
		haspCmd.Run()

		// Give it some time to connect to the remote license server and get available licenses
		time.Sleep(10 * time.Second)
		features, err = GetSentinelFeatures()
		if err != nil {
			log.Fatal(err)
		}
	}

	// If running CYME gui, check to see if the CYME_NETWORK_EDITOR is available
	// If running CYME python script, check to see if the CYME_PYTHON_SCRIPTING_DEVELOPER is available
	// Any of these features not available might mean that the local license manager wasn't able to connect to the remove server

	// TODO try to autodetect this path if not given
	wineDirPath := "/usr/workspace/wsb/cyme/wine-3.0.2-32"

	// Add WINE libraries to LD_LIBRARY_PATH
	wineLibToAdd := wineDirPath + "/lib"
	wineLibToAdd += string(os.PathListSeparator) + wineDirPath + "/usr/lib"
	wineLibToAdd += string(os.PathListSeparator) + wineDirPath + "/usr/lib/dri"
	wineLibPath := os.ExpandEnv("${LD_LIBRARY_PATH}" + string(os.PathListSeparator) + wineLibToAdd)
	os.Setenv("LD_LIBRARY_PATH", wineLibPath)

	// Add WINE binaries to PATH
	wineBinToAdd := wineDirPath + "/usr/bin"
	wineBinPath := os.ExpandEnv("${PATH}" + string(os.PathListSeparator) + wineBinToAdd)
	os.Setenv("PATH", wineBinPath)

	fmt.Println(wineBinPath)
	// Set LIBGL_DRIVERS_PATH
	os.Setenv("LIBGL_DRIVERS_PATH", wineDirPath+"/usr/lib/dri")

	// Launch CYME gui in WINE, or run Python in Wine with the given script
	// TODO try to autodetect wine32_prefix directory if not given (or not already set)
	// TODO make a copy/symlinks of wine32_prefix
	// cd wine32_prefix/drive_c/Program\ Files/CYME/CYME
	// WINEPREFIX=$HOME/test_wine_symlink wine Cyme.exe

	// create tmp directory for wine prefix (wine is picky about ownership of directories)
	winePrefixTmpDir, err := ioutil.TempDir("", "wine32_prefix")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(winePrefixTmpDir)
	//defer os.RemoveAll(winePrefixTmpDir)

	// symlink contents of wine prefix folder to our temporary one
	files, err := ioutil.ReadDir(winePrefix)
	if err != nil {
		log.Fatal(err)
	}
	for _, file := range files {
		fmt.Println(file.Name())
		target := filepath.Join(winePrefix, file.Name())
		symlink := filepath.Join(winePrefixTmpDir, file.Name())
		os.Symlink(target, symlink)
	}

	os.Setenv("WINEPREFIX", winePrefixTmpDir)

	// If we are the last cyme-launcher running, we should shut down hasplmd before quitting (get system process list and search names?)
	// Or if we started up hasplmd, then we should kill it -- that might be better, since this launcher could be used standalone most of the time
	// batch job with several tasks on the same node may be better off just spawning its own hasplmd first

	// ProductID - 1000 = FeatureID
	fmt.Printf("Feature available: %v\n", CheckFeatureAvailable(features, CYME_NETWORK_EDITOR, CYME_NETWORK_EDITOR-1000))
	feature := FindFeature(features, CYME_NETWORK_EDITOR, CYME_NETWORK_EDITOR-1000)
	if feature != nil {
		fmt.Printf("Licenses used: %v/%v\n", feature.LoginCount, feature.LoginLimit)
	}

	if feature == nil || feature.LoginCount >= feature.LoginLimit {
		log.Fatal("CYME Network Editor not available")
	}

	//os.Setenv("WINEDEBUG","+relay")
	var wineCmd *exec.Cmd
	if pythonScript != "" {
		wineCmd = exec.Command("wine", "python", pythonScript, strings.Join(flag.Args(), " "))
	} else {
		wineCmd = exec.Command("wine", cymePath+"\\Cyme.exe", strings.Join(flag.Args(), " "))
	}
	wineCmd.Stdout = os.Stdout
	wineCmd.Stderr = os.Stderr
	err = wineCmd.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error from WINE command:\n%v\n", err)
		os.Exit(1)
	}
}

// default config file settings for hasp lm
// eventually this should be turned into a struct that gets serialized into a ini file that matches this
var haspConfigContents = `
[SERVER]
name = %v
idle_session_timeout_mins = 720
pagerefresh = 3
linesperpage = 20
ACCremote = 0
enablehaspc2v = 0
old_files_delete_days = 90

enabledetach = 0
reservedseats = 0
reservedpercent = 0
detachmaxdays = 14
commuter_delete_days = 7
disable_um = 0

requestlog = 0
loglocal = 0
logremote = 0
logadmin = 0
errorlog = 0
rotatelogs = 0
access_log_maxsize = 0 ;kB
error_log_maxsize = 0 ;kB
zip_logs_days = 0
delete_logs_days = 0
pidfile = 0
passacc = 0

accessfromremote = 1
accesstoremote = 1
bind_local_only = 0  ; 0=all adapters, 1=localhost only

[UPDATE]
download_url = sentinelcustomer.gemalto.com/Sentinel/LanguagePacks/
update_host = www3.safenet-inc.com
language_url = /hasp/language_packs/end-user/

[REMOTE]
broadcastsearch = 1
aggressive = 1
serversearchinterval = 30
serveraddr = %v

[ACCESS]

[USERS]

[VENDORS]

[EMS]
emsurl = http://localhost:8080
emsurl = http://127.0.0.1:8080

[LOGPARAMETERS]
text = {timestamp} {clientaddr}:{clientport} {clientid} {method} {url} {function}({functionparams}) result({statuscode}){newline}
`

/* Helper functions - shouldn't need changing unless Sentinel license manager changes */

// CYME Product IDs, subtract 1000 for feature IDs (Vendor 41297)
// 1003 - CYMDIST
// 1019 - CYME Substation Modeling
// 1056 - CYME Energy Profile Manager
// 1066 - CYME - Long Term Dynamics
// 1080 - CYME Integration Capacity Analysis
// 1100 - CYME Network Editor
// 1112 - CYME Python Scripting - Developer
// 1114 - CYME Low Voltage Distribution Network
// 10000 - CYME International TD Inc
const (
	CYMDIST                               = 1003
	CYME_SUBSTATION_MODELING              = 1019
	CYME_ENERGY_PROFILE_MANAGER           = 1056
	CYME_LONG_TERM_DYNAMICS               = 1066
	CYME_INTEGRATION_CAPACITY_ANALYSIS    = 1080
	CYME_NETWORK_EDITOR                   = 1100
	CYME_PYTHON_SCRIPTING_DEVELOPER       = 1112
	CYME_LOW_VOLTAGE_DISTRIBUTION_NETWORK = 1114
	CYME_INTERNATIONAL_TD_INC             = 10000
)

type SentinelFeature struct {
	Index       int    `json:"ndx,string"`
	VendorId    int    `json:"ven,string"`
	HaspName    string `json:"haspname"`
	HaspId      int    `json:"haspid,string"`
	File        string `json:"file"`
	Ip          string `json:"ip"`
	FeatureId   int    `json:"fid,string"`
	FeatureName string `json:"fn"`
	Location    string `json:"loc"`
	IsLocal     int    `json:"isloc,string"`
	Access      string `json:"acc"`
	Counting    string `json:"cnt"` // Station/Process
	LoginCount  int    `json:"logc,string"`
	LoginLimit  int    `json:"logl,string"`
	// logp
	Detachable    int `json:"det,string"` // bool
	DetachedCount int `json:"detc,string"`
	Locked        int `json:"locked,string"` // bool
	// dis, ex
	Unusable     int    `json:"unusable,string"` // bool
	Restrictions string `json:"lic"`
	SessionCount int    `json:"sesc,string"`
	ProductName  string `json:"prname"`
	ProductId    int    `json:"prid,string"`
	Type         string `json:"typ"`
}

func (f *SentinelFeature) Available() bool {
	if f.LoginLimit == -1 {
		return true
	}
	return f.LoginCount < f.LoginLimit
}

func CheckFeatureAvailable(features []SentinelFeature, productId, featureId int) bool {
	for _, v := range features {
		if v.ProductId == productId && v.FeatureId == featureId {
			return v.Available()
		}
	}
	return false
}

func FindFeature(features []SentinelFeature, productId, featureId int) *SentinelFeature {
	for _, v := range features {
		if v.ProductId == productId && v.FeatureId == featureId {
			return &v
		}
	}
	return nil
}

func fetchUrlJson(url string) ([][]byte, error) {
	r, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer r.Body.Close()

	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}

	// The comments in the response returned aren't part of JSON and don't work with json.Unmarshal
	// Remove /*JSON:____*/ comment
	arrStart := bytes.Index(b, []byte("*/"))
	if arrStart != -1 {
		b = b[arrStart+2:]
	}
	// Remove any comments after JSON objects
	arrEnd := bytes.LastIndex(b, []byte("/*"))
	if arrEnd != -1 {
		b = b[:arrEnd]
	}
	// Trim whitespace, split JSON objects, and remove trailing "," for unmarshaling
	b = bytes.TrimSpace(b)
	jsonObjects := bytes.SplitAfter(b, []byte("},"))

	for i, v := range jsonObjects {
		jsonObjects[i] = bytes.TrimSpace(bytes.TrimSuffix(v, []byte(",")))
	}
	return jsonObjects, nil
}

func GetSentinelFeatures() ([]SentinelFeature, error) {
	url := fmt.Sprintf("http://localhost:1947/_int_/tab_feat.html?haspid=0&featureid=-1&vendorid=0&productid=0&filterfrom=1&filterto=20&timestamp=%v?", time.Now().UnixNano()/1000000)
	jsonObjects, err := fetchUrlJson(url)
	if err != nil {
		return nil, err
	}

	var f []SentinelFeature
	for _, v := range jsonObjects {
		var tmp SentinelFeature
		err = json.Unmarshal(v, &tmp)
		if err != nil {
			return f, nil
		}
		f = append(f, tmp)
	}
	return f, nil
}
