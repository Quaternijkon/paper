# Pack

How to pack the binary file.

    cargo build --release --target x86_64-unknown-linux-gnu

    cd target/x86_64-unknown-linux-gnu/release

    tar -czvf my_project-x86_64-unknown-linux-gnu.tar.gz my_project

