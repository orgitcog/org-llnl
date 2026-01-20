// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.offsetmanager;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.apache.commons.lang3.SerializationUtils;
import org.apache.curator.framework.CuratorFramework;
import org.apache.curator.framework.CuratorFrameworkFactory;
import org.apache.curator.framework.state.ConnectionState;
import org.apache.curator.framework.state.ConnectionStateListener;
import org.apache.curator.retry.RetryForever;

import static gov.llnl.sonar.kafka.connect.offsetmanager.FileOffsetManager.makeOffsetPath;

public class CLI {

    @Parameter(names={"--zk-host", "-h"})
    String zooKeeperHost = "localhost";

    @Parameter(names={"--zk-port", "-p"})
    int zooKeeperPort = 2181;

    @Parameter(names={"--ingest-dir", "-d"})
    String ingestDir;

    @Parameter(names={"--ingest-file", "-f"})
    String ingestFile;

    public static void main(String... argv) {
        CLI cli = new CLI();
        JCommander.newBuilder()
                .addObject(cli)
                .build()
                .parse(argv);

        try {
            cli.run();
        } catch (Exception e) {
            System.err.println(e.toString());
        }
    }

    public void run() throws Exception {
        CuratorFramework client = CuratorFrameworkFactory.newClient(
                zooKeeperHost + ":" + zooKeeperPort,
                10000,
                10000,
                new RetryForever(1000));

        client.start();

        FileOffset fileOffset;
        String actualFileOffsetPath = makeOffsetPath(ingestDir, ingestFile);

        byte[] fileOffsetBytes = client.getData().forPath(actualFileOffsetPath);
        fileOffset = SerializationUtils.deserialize(fileOffsetBytes);

        System.out.println(fileOffset.toString());
    }
}
