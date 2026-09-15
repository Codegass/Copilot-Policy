"""Lifecycle compatibility for the pinned I2RT ac096928 driver."""

import threading


def close_driver(driver):
    # Stop the command producer before stopping the CAN consumer. Native
    # MotorChainRobot.close() only joins the producer, then closes the socket
    # while DMChainCanInterface may still be sending a motor batch.
    driver._stop_event.set()
    driver._server_thread.join()
    chain = driver.motor_chain
    # This SDK keeps its worker only in a local variable in start_thread().
    # Identify the bound target by its owner, never by a shared thread name;
    # the other arm's CAN worker must remain independent.
    workers = [
        thread for thread in threading.enumerate()
        if getattr(getattr(thread, "_target", None), "__self__", None) is chain
    ]
    chain.running = False
    for worker in workers:
        worker.join()
    driver.close()
