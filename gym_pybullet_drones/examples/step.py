

def step(self,
         action
         ):
    """Advances the environment by one simulation step.
    Parameters
    ----------
    action : ndarray | dict[..]
        The input action for one or more drones, translated into RPMs by
        the specific implementation of `_preprocessAction()` in each subclass.
    Returns
    -------
    ndarray | dict[..]
        The step's observation, check the specific implementation of `_computeObs()`
        in each subclass for its format.
    float | dict[..]
        The step's reward value(s), check the specific implementation of `_computeReward()`
        in each subclass for its format.
    bool | dict[..]
        Whether the current episode is over, check the specific implementation of `_computeTerminated()`
        in each subclass for its format.
    bool | dict[..]
        Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
        in each subclass for its format.
    bool | dict[..]
        Whether the current episode is truncated, always false.
    dict[..]
        Additional information as a dictionary, check the specific implementation of `_computeInfo()`
        in each subclass for its format.
    """
    #### Save PNG video frames if RECORD=True and GUI=False ####
    if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                 height=self.VID_HEIGHT,
                                                 shadow=1,
                                                 viewMatrix=self.CAM_VIEW,
                                                 projectionMatrix=self.CAM_PRO,
                                                 renderer=p.ER_TINY_RENDERER,
                                                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                 physicsClientId=self.CLIENT
                                                 )
        (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
        #### Save the depth or segmentation view instead #######
        # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
        # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
        # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
        # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
        self.FRAME_NUM += 1
        if self.VISION_ATTR:
            for i in range(self.NUM_DRONES):
                self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                #### Printing observation to PNG frames example ############
                self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                img_input=self.rgb[i],
                                path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                )
    #### Read the GUI's input parameters #######################
    if self.GUI and self.USER_DEBUG:
        current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
        if current_input_switch > self.last_input_switch:
            self.last_input_switch = current_input_switch
            self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
    if self.USE_GUI_RPM:
        for i in range(4):
            self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
        clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
        if self.step_counter%(self.PYB_FREQ/2) == 0:
            self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                      textPosition=[0, 0, 0],
                                                      textColorRGB=[1, 0, 0],
                                                      lifeTime=1,
                                                      textSize=2,
                                                      parentObjectUniqueId=self.DRONE_IDS[i],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                      physicsClientId=self.CLIENT
                                                      ) for i in range(self.NUM_DRONES)]
    #### Save, preprocess, and clip the action to the max. RPM #
    else:
        clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
    #### Repeat for as many as the aggregate physics steps #####
    for _ in range(self.PYB_STEPS_PER_CTRL):
        #### Update and store the drones kinematic info for certain
        #### Between aggregate steps for certain types of update ###
        if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
            self._updateAndStoreKinematicInformation()
        #### Step the simulation using the desired physics update ##
        for i in range (self.NUM_DRONES):
            if self.PHYSICS == Physics.PYB:
                self._physics(clipped_action[i, :], i)
            elif self.PHYSICS == Physics.DYN:
                self._dynamics(clipped_action[i, :], i)
            elif self.PHYSICS == Physics.PYB_GND:
                self._physics(clipped_action[i, :], i)
                self._groundEffect(clipped_action[i, :], i)
            elif self.PHYSICS == Physics.PYB_DRAG:
                self._physics(clipped_action[i, :], i)
                self._drag(self.last_clipped_action[i, :], i)
            elif self.PHYSICS == Physics.PYB_DW:
                self._physics(clipped_action[i, :], i)
                self._downwash(i)
            elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                self._physics(clipped_action[i, :], i)
                self._groundEffect(clipped_action[i, :], i)
                self._drag(self.last_clipped_action[i, :], i)
                self._downwash(i)
        #### PyBullet computes the new state, unless Physics.DYN ###
        if self.PHYSICS != Physics.DYN:
            p.stepSimulation(physicsClientId=self.CLIENT)
        #### Save the last applied action (e.g. to compute drag) ###
        self.last_clipped_action = clipped_action
    #### Update and store the drones kinematic information #####
    self._updateAndStoreKinematicInformation()
    #### Prepare the return values #############################
    obs = self._computeObs()
    reward = self._computeReward()
    terminated = self._computeTerminated()
    truncated = self._computeTruncated()
    info = self._computeInfo()
    #### Advance the step counter ##############################
    self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
    return obs, reward, terminated, truncated, info
