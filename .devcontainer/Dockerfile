# Start from the correct Humble image
FROM osrf/ros:jazzy-desktop

ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Delete user if it exists in container (e.g Ubuntu Noble: ubuntu)
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3-pip
ENV SHELL /bin/bash
RUN apt-get install -y \
    libglu1-mesa \
    libsm6 \
    libxrender1 \
    libfontconfig1 \
    libxext6 \
    xz-utils \
    # Clean up APT caches to reduce image size
    && rm -rf /var/lib/apt/lists/*
# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

COPY CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04.tar.xz /tmp/

RUN mkdir -p /home/${USERNAME}/software_dir
RUN tar xJf /tmp/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04.tar.xz -C /home/$USERNAME/software_dir/ \
&& rm /tmp/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04.tar.xz

RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/software_dir/
# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
CMD ["/bin/bash"]

# Option B (Recommended for final image): Start CoppeliaSim immediately
# CMD ["/home/gokulkishore/software_dir/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu24_04/coppeliaSim.sh"]


# to allow GUI of container app in our main os :
#gokulkishore@valuable:~$ xhost +local:root
#non-network local connections being added to access control list








