// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.offsetmanager;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.SerializationUtils;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.recipes.locks.InterProcessLock;
import org.apache.curator.framework.recipes.locks.InterProcessMutex;
import org.apache.curator.framework.state.ConnectionState;
import org.apache.curator.framework.state.ConnectionStateListener;
import org.apache.curator.retry.RetryForever;
import org.apache.curator.utils.ZKPaths;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Object for managing distributed file offsets for concurrent ingestion.
 * Uses Apache Curator (which uses Zookeeper) to enforce global, process/thread wide locks on FileOffset objects.
 */

@Slf4j
public class FileOffsetManager {

    private Long threadID;

    private String fileOffsetBasePath;

    private CuratorFramework client;

    private InterProcessLock lock;

    final static Long offsetFileTTL = 30L*1000L;

    final static String LOCKS_SUBPATH = "locks";
    final static String OFFSETS_SUBPATH = "offsets";

    /**
     * Constructor that creates a Curator client, initializes an InterProcessLock, and Zookeeper nodes
     * for a provided directory.
     * For a directory /usr/foo, will create a mirror Zookeeper directory /usr/foo that manages locks to
     * all files in /usr/foo.
     *
     * @param zooKeeperHost Zookeeper host to connect to (e.g. localhost)
     * @param zooKeeperPort Zookeeper port to connect to (e.g. 2181)
     * @param fileOffsetBasePath Provided directory to create locks for
     * @param reset Whether to initialize the directory as empty (reset)
     * @throws Exception From Curator
     */
    public FileOffsetManager(String zooKeeperHost, String zooKeeperPort, String fileOffsetBasePath, boolean reset) throws Exception {
        this.threadID = Thread.currentThread().getId();
        this.fileOffsetBasePath = fileOffsetBasePath;

        //log.debug("Thread {}: Initializing zookeeper connection", threadID);

        client = CuratorFrameworkFactory.newClient(
                zooKeeperHost + ":" + zooKeeperPort,
                10000,
                10000,
                new RetryForever(1000));

        FileOffsetManager thisRef = this;
        client.getConnectionStateListenable().addListener(new ConnectionStateListener() {
            @Override
            public void stateChanged(CuratorFramework client, ConnectionState newState) {
                if (!newState.isConnected()) {
                    log.warn("Thread {}: Curator state changed to {} with contents: {}", threadID, newState.toString(), thisRef.toString());
                }
            }
        });
        client.start();

        //log.debug("Thread {}: Zookeeper connection initialized", threadID);

        if (reset) {
            reset();
        }

        //log.debug("Thread {}: Initializing lock", threadID);
        lock = new InterProcessMutex(client, ZKPaths.makePath(fileOffsetBasePath, LOCKS_SUBPATH));
        //log.debug("Thread {}: Lock initialized", threadID);
    }

    /**
     * Delete all entries in the file byteOffset directory (offsets/lock) and create new ones.
     *
     * @throws Exception From Curator
     */
    private void reset() throws Exception {

        //log.debug("Thread {}: Checking for file getByteOffset base path {}", threadID, fileOffsetBasePath);

        if (client.checkExists().creatingParentContainersIfNeeded().forPath(fileOffsetBasePath) != null) {
            //log.debug("Thread {}: Deleting previous file getByteOffset base path {}", threadID, fileOffsetBasePath);
            client.delete().deletingChildrenIfNeeded().forPath(fileOffsetBasePath);
        }

        //log.debug("Thread {}: Creating file getByteOffset base path {}, locks, and offsets", threadID, fileOffsetBasePath);

        client.create().creatingParentContainersIfNeeded().forPath(fileOffsetBasePath);
        client.create().forPath(ZKPaths.makePath(fileOffsetBasePath, LOCKS_SUBPATH));
        client.create().forPath(ZKPaths.makePath(fileOffsetBasePath, OFFSETS_SUBPATH));
    }

    @Override
    public String toString() {
        final String lockString;

        if (lock != null) {
            lockString = String.valueOf(lock.isAcquiredInThisProcess());
        } else {
            lockString = "null";
        }

        return "FileOffsetManager(Path=" + fileOffsetBasePath + ", Locked=" + lockString + ")";
    }

    /**
     * Provided a file path string under the file byteOffset directory,
     * creates a Zookeeper node name under "offsets" within the file byteOffset directory
     * e.g. if byteOffset directory is "/usr/foo" and filePath is "/usr/foo/bar/file.txt",
     * creates Zookeeper node name "/usr/foo/offsets/bar/file.txt".
     *
     * @param filePath File path string for which to create an offsets Zookeeper node
     * @return Zookeeper path for new offsets node.
     */
    private String makeOffsetPath(String filePath) {
        return makeOffsetPath(fileOffsetBasePath, filePath);
    }

    public static String makeOffsetPath(String fileOffsetBasePath, String filePath) {
        Path relativePath = Paths.get(fileOffsetBasePath).relativize(Paths.get(filePath)).normalize();
        return ZKPaths.makePath(fileOffsetBasePath, OFFSETS_SUBPATH, relativePath.toString());
    }

    /**
     * Uploads the provided FileOffset to the appropriate Zookeeper node managing the provided file byteOffset path.
     * Effectively updates the status of the file byteOffset.
     *
     * @param fileOffsetPath The path of the file for which to update the file byteOffset
     * @param fileOffset The new file byteOffset
     * @throws Exception From Curator
     */
    public void uploadFileOffset(String fileOffsetPath, FileOffset fileOffset) throws Exception {

        String actualFileOffsetPath = makeOffsetPath(fileOffsetPath);

        //log.debug("Thread {}: Uploading file getByteOffset {}: {}", threadID, actualFileOffsetPath, fileOffset.toString());

        client.create().orSetData().forPath(actualFileOffsetPath, SerializationUtils.serialize(fileOffset));

        //log.info("Thread {}: Uploaded file offset {}: {}", threadID, actualFileOffsetPath, fileOffset.toString());
    }

    /**
     * Downloads the FileOffset for a provided file, initializing a new one if it doesn't exist.
     * IMPORTANT: This function is not atomic and therefore must be called inside lock()/unlock()!
     *
     * @param fileOffsetPath The file for which to download the file byteOffset
     * @return The file byteOffset
     * @throws Exception From Curator
     */
    public FileOffset downloadFileOffsetWithLock(String fileOffsetPath) throws Exception {

        FileOffset fileOffset;
        String actualFileOffsetPath = makeOffsetPath(fileOffsetPath);

        //log.debug("Thread {}: Downloading file getByteOffset if exists: {}", threadID, actualFileOffsetPath);

        if (client.checkExists().creatingParentContainersIfNeeded().forPath(actualFileOffsetPath) == null) {
            //log.debug("Thread {}: File getByteOffset does not exist, creating and locking it: {} ", threadID, actualFileOffsetPath);
            fileOffset = new FileOffset(0L, 0L, true, false);
            client.create().forPath(actualFileOffsetPath, SerializationUtils.serialize(fileOffset));
        } else {
            //log.debug("Thread {}: File getByteOffset exists, getting it: {} ", threadID, actualFileOffsetPath);
            byte[] fileOffsetBytes = client.getData().forPath(actualFileOffsetPath);
            fileOffset = SerializationUtils.deserialize(fileOffsetBytes);
            if (fileOffset.locked || fileOffset.completed) {
                fileOffset = null;
            } else {
                fileOffset.locked = true;
                client.setData().forPath(actualFileOffsetPath, SerializationUtils.serialize(fileOffset));
            }
        }

        //log.info("Thread {}: Downloaded file offset {}: {}", threadID, actualFileOffsetPath, fileOffset);

        return fileOffset;
    }

    /**
     * Downloads the file offset for a given file if it exists (does not create/modify anything)
     * @param fileOffsetPath The file offset to download
     * @return The deserialized FileOffset object
     * @throws Exception from Curator
     */
    public FileOffset downloadFileOffsetWithoutLock(String fileOffsetPath) throws Exception {

        FileOffset fileOffset;
        String actualFileOffsetPath = makeOffsetPath(fileOffsetPath);

        byte[] fileOffsetBytes = client.getData().forPath(actualFileOffsetPath);
        fileOffset = SerializationUtils.deserialize(fileOffsetBytes);

        return fileOffset;
    }

    /**
     * Obtains the global (process/thread-wide) lock for this FileOffsetManager instance.
     */
    public void lock() {
        try {
            //log.debug("Thread {}: Acquiring lock for {}", threadID, fileOffsetBasePath);
            lock.acquire();
            //log.debug("Thread {}: Acquired lock for {}", threadID, fileOffsetBasePath);
        } catch (Exception e) {
            log.error("Thread {}: {}", threadID, e);
        }
    }

    /**
     * Releases the global (process/thread-wide) lock for this FileOffsetManager instance.
     */
    public void unlock() {
        if(lock.isAcquiredInThisProcess()) {
            try {
                //log.debug("Thread {}: Releasing lock for {}", threadID, fileOffsetBasePath);
                lock.release();
                //log.debug("Thread {}: Released lock for {}", threadID, fileOffsetBasePath);
            } catch (Exception e) {
                log.error("Thread {}: {}", threadID, e);
            }
        }
    }

    public void close() {
        unlock();
        //log.debug("Thread {}: Closing zookeeper client", threadID);
        client.close();
        //log.debug("Thread {}: Closed zookeeper client", threadID);
    }
}
